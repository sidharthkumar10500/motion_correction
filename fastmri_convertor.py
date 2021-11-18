'''
This script converts the fastmri dataset into a single coil data with mvue combined images, 
furthermore the output dimensions are [H,W,Slices]
The outputs are saved in a torch format with all the header information, that might be usefull later on. 
'''
import numpy as np
from tqdm import tqdm
import os, torch, h5py, argparse
from bart import bart
import cfl 
from typing import Sequence
import xml.etree.ElementTree as etree

parser = argparse.ArgumentParser(description='Reading args for converting the fastmri dataset')
parser.add_argument('-g','--global_dir', type=str, default='/csiNAS2/slow/mridata/fastmri_brain/multicoil_train', metavar='', help = 'Choose the data containing directory')
args = parser.parse_args()

def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)



if __name__ == "__main__":
    global_dir = args.global_dir
    os.chdir(global_dir)
    files  = os.listdir(global_dir)
    np.random.seed(22) 
    index = np.random.shuffle(np.arange(len(files)))

    for file_name in tqdm(files[0:500]):#iterating over all the files in the global directory
        #read h5 file
        a_file = h5py.File(file_name, "r")
        kspace_data = np.array(a_file.get('kspace'))
        Rss_recon_image = np.array(a_file.get('reconstruction_rss'))
        file_data = np.array(a_file.get('ismrmrd_header'))
        a_file.close()
        # close h5 file

        # reader the header data
        contents = h5py.File(file_name, 'r')
        header   = contents['ismrmrd_header']
        et_root  = etree.fromstring(header[()])

        # Get some stuff
        enc = ["encoding", "encodedSpace", "matrixSize"]
        enc_size = (
            int(et_query(et_root, enc + ["x"])),
            int(et_query(et_root, enc + ["y"])),
            int(et_query(et_root, enc + ["z"])),
        )

        rec = ["encoding", "reconSpace", "matrixSize"]
        recon_size = (
            int(et_query(et_root, rec + ["x"])),
            int(et_query(et_root, rec + ["y"])),
            int(et_query(et_root, rec + ["z"])),
        )

        trajectory = et_query(et_root, ['encoding', 'trajectory'])

        # measurementInformation {}
        protocolName = et_query(et_root, ['measurementInformation', 'protocolName'])
        patientPosition = et_query(et_root, ['measurementInformation', 'patientPosition'])

        # sequenceParameters {}
        TR = et_query(et_root, ['sequenceParameters', 'TR'])
        TE = et_query(et_root, ['sequenceParameters', 'TE'])
        TI = et_query(et_root, ['sequenceParameters', 'TI'])
        flipAngle_deg = et_query(et_root, ['sequenceParameters', 'flipAngle_deg'])
        sequence_type = et_query(et_root, ['sequenceParameters', 'sequence_type'])

        # acquisitionSystemInformation {}
        institutionName = et_query(et_root, ['acquisitionSystemInformation', 'institutionName'])
        systemVendor    = et_query(et_root, ['acquisitionSystemInformation', 'systemVendor'])
        systemModel     = et_query(et_root, ['acquisitionSystemInformation', 'systemModel'])
        systemFieldStrength_T = et_query(et_root, ['acquisitionSystemInformation', 'systemFieldStrength_T'])
        receiverChannels = et_query(et_root, ['acquisitionSystemInformation', 'receiverChannels'])
        # reading header data is over

        # bring the kspace in the required format
        kspace_data = np.moveaxis(kspace_data, [2, 3], [0, 1])#swap axises to bring X and  Y in first 2 dimensions
        # print('kspace_data.shape:-',kspace_data.shape) #slices are at 2 and coils are at 3
        kspace_data = np.flipud(kspace_data)
        NUM_slices = kspace_data.shape[2]
        NUM_coils  = kspace_data.shape[3]

        mvue_recon = np.zeros((320,320,NUM_slices),dtype=complex)

        # estimate coil sensitivites and do mvue recon
        OMP_NUM_THREADS = "20" 
        os.environ["OMP_NUM_THREADS"] = "1"
        for slice_index in range(NUM_slices):
            coil_sens = bart(1, 'ecalib -a -m 1 -d 0', kspace_data[None,:,:,slice_index,:])#estimate coil sens
            
            xyspace = bart(1,'fft -i -u 3', kspace_data[:,:,slice_index,:])
            mvue = np.sum(xyspace* np.conj(coil_sens.squeeze()), axis = 2)
            mvue_recon[:,:,slice_index] = bart(1,'resize -c 0 {} 1 {}'.format(320,320), mvue)

        kspace_single_coil = bart(1,'fft -u 3', mvue_recon)#also calculate the kspace


        # save the torch tensor with all the header information
        local_dir = '/home/sidharth/sid_notebooks/motion_correction/training_data/'
        tosave_weights = local_dir + file_name +'.pt' 
        torch.save({
            'kspace_single_coil': kspace_single_coil,
            'mvue_recon': mvue_recon,
            'trajectory': trajectory,
            'protocolName':protocolName,
            'patientPosition': patientPosition,
            'TR': TR,
            'TE': TE,
            'TI': TI,
            'flipAngle_deg': flipAngle_deg,
            'sequence_type': sequence_type,
            'institutionName':institutionName,
            'systemVendor':systemVendor,
            'systemModel':systemModel,
            'systemFieldStrength_T': systemFieldStrength_T,
            'receiverChannels':receiverChannels}, tosave_weights)
