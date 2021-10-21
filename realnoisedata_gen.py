# Aim of this file is to generate good .h5 files from the 
# real noise dataset
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import utils
from tqdm import tqdm
import os
import h5py
import glob
import ismrmrd
from ismrmrdtools import show, transform
import ismrmrd.xsd
import argparse

parser = argparse.ArgumentParser(description='Reading args for running the deep network training')
parser.add_argument('-m','--mini_contrast', type=str, default='t2', metavar='', help = 'contrast to choose from {t1,t2}') #positional argument
parser.add_argument('-t','--target_scan', type=int, default=1, metavar='', help = 'Choose scan ID from 1 to 5') #optional argument
parser.add_argument('-g','--global_dir', type=str, default='', metavar='', help = 'Choose the data containing directory')
args = parser.parse_args()

global_dir = '/csiNAS/brett/RealNoiseMRI/mnt/mocodata1/MoCoChallenge/MedNeurIPS/Download_Validation_Data/'
global_dir = '/home/sidharth/'
mini_contrast   = args.mini_contrast
target_scan     =  args.target_scan # From 1 to 5

# Target files
nod_file   = global_dir + 'VAL_0%d/Scan_nod_%s.h5' % (
    target_scan, mini_contrast)

still_file = global_dir + 'VAL_0%d/Scan_still_%s.h5' % (
    target_scan, mini_contrast)

def h5_generator(file):
    dset_nod = ismrmrd.Dataset(file, 'dataset', create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dset_nod.read_xml_header())
    enc = header.encoding[0]

    # Matrix size
    eNx = enc.encodedSpace.matrixSize.x
    eNy = enc.encodedSpace.matrixSize.y
    eNz = enc.encodedSpace.matrixSize.z
    rNx = enc.reconSpace.matrixSize.x
    rNy = enc.reconSpace.matrixSize.y
    rNz = enc.reconSpace.matrixSize.z

    # Field of View
    eFOVx = enc.encodedSpace.fieldOfView_mm.x
    eFOVy = enc.encodedSpace.fieldOfView_mm.y
    eFOVz = enc.encodedSpace.fieldOfView_mm.z
    rFOVx = enc.reconSpace.fieldOfView_mm.x
    rFOVy = enc.reconSpace.fieldOfView_mm.y
    rFOVz = enc.reconSpace.fieldOfView_mm.z

    # Number of Slices, Reps, Contrasts, etc.
    ncoils = header.acquisitionSystemInformation.receiverChannels

    if enc.encodingLimits.slice != None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        nslices = 1

    if enc.encodingLimits.repetition != None:
        nreps = enc.encodingLimits.repetition.maximum + 1
    else:
        nreps = 1

    if enc.encodingLimits.contrast != None:
        ncontrasts = enc.encodingLimits.contrast.maximum + 1
    else:
        ncontrasts = 1

    # TODO loop through the acquisitions looking for noise scans
    firstacq=0
    for acqnum in range(dset_nod.number_of_acquisitions()):
        acq = dset_nod.read_acquisition(acqnum)
        
        # TODO: Currently ignoring noise scans
        if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            print("Found noise scan at acq ", acqnum)
            continue
        else:
            firstacq = acqnum
            print("Imaging acquisition starts acq ", acqnum)
            break


    # Initialiaze a storage array
    all_data = np.zeros((nreps, ncontrasts, nslices, ncoils, eNz, eNy, rNx),
                        dtype=np.complex64)

    print(all_data.shape)
    # Loop through the rest of the acquisitions and stuff
    for acqnum in tqdm(range(firstacq,dset_nod.number_of_acquisitions())):
        acq = dset_nod.read_acquisition(acqnum)

        # TODO: this is where we would apply noise pre-whitening
        
        # Remove oversampling if needed
        if eNx != rNx:
            xline = transform.transform_kspace_to_image(acq.data, [1])
            x0 = int((eNx - rNx) / 2)
            x1 = int((eNx - rNx) / 2 + rNx)
            xline = xline[:,x0:x1]
            acq.resize(rNx,acq.active_channels,acq.trajectory_dimensions)
            acq.center_sample = int(rNx/2)
            # need to use the [:] notation here to fill the data
            acq.data[:] = transform.transform_image_to_kspace(xline, [1])
    
        # Stuff into the buffer
        rep = acq.idx.repetition
        contrast = acq.idx.contrast
        slice = acq.idx.slice
        y = acq.idx.kspace_encode_step_1
        z = acq.idx.kspace_encode_step_2
        all_data[rep, contrast, slice, :, z, y, :] = acq.data
        return all_data


nod_data = h5_generator(nod_file)
still_data = h5_generator(still_file)

nod_file_name = 'subject_{}_contrast_{}_nod.h5'.format(target_scan,mini_contrast)
still_file_name = 'subject_{}_contrast_{}_still.h5'.format(target_scan,mini_contrast)
with h5py.File(nod_file_name, 'w') as F:
   F.create_dataset('kspace_data', data=nod_data)
with h5py.File(still_file_name, 'w') as F:
   F.create_dataset('kspace_data', data=still_data)
