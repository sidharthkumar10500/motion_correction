import numpy as np
import matplotlib.pyplot as plt
import os, sys
# os.environ["TOOLBOX_PATH"] = "/home/blevac/misc/HW/bart-0.6.00"  ## change to YOUR path
# sys.path.append("/home/blevac/misc/HW/bart-0.6.00/python")        ## change to YOUR path
from bart import bart#required for importing bart
from tqdm.auto import tqdm

import os, torch, argparse, h5py, glob


os.chdir('/home/sidharth/sid_notebooks/MDME_DL_datacreation')#change current working directory
import functions_def as function_defs
from scipy import ndimage
import sigpy as sp
os.chdir('/home/sidharth/sid_notebooks/')#change current working directory

# argument parser, change defaults using terminal inputs
parser = argparse.ArgumentParser(description='Reading args for running the deep network training')
parser.add_argument('-sb','--subject', type=int, default=24, metavar='', help = 'Subject index whose data need to be reconstructed') #positional argument
parser.add_argument('-sl','--slice', type=int, default=24, metavar='', help = 'slice index of the above subject') #positional argument
parser.add_argument('-etl','--etl', type=int, default=8, metavar='', help = 'echo train length') #positional argument
parser.add_argument('-te','--TE_effective', type=int, default=110, metavar='', help = 'effective echo length') #positional argument
parser.add_argument('-teb','--TE_bw_echoes', type=int, default=10, metavar='', help = 'time between different echoes in a single TR') #positional argument


def signal_decay_sim(T2, PD, TE, mask):
    #a single signal value at a particulr TE
    img_out = np.zeros(mask.shape).astype(complex)
    T2_values = T2[np.nonzero(mask)] 
    PD_values = PD[np.nonzero(mask)] 
    
    img_out[np.nonzero(mask)] = PD_values*np.exp(-TE/T2_values)

    return img_out

def Echo_image_gen(T2, PD, mask, TEs):
    # echo images at all the possible TEs
    Echo_imgs = []
    for TE in TEs:
        Echo_img = signal_decay_sim(T2 = T2, PD = PD, TE = TE, mask = mask)
        Echo_imgs.append(Echo_img)
        
    return np.asarray(Echo_imgs)

def motion_gen(Echo_images, angles):
    # generating motion for all the echo images, angles have the length of number of TRs
    motion_contrast_imgs = [[] for i in range(Echo_images.shape[0]) ]
    for shot_num in range(len(angles)):
        
        for echo_num in range(Echo_images.shape[0]):
            motion_img = ndimage.rotate(input=Echo_images[echo_num], angle=angles[shot_num],
                        axes=(1, 0), reshape=False, output=None, order=3,
                        mode='constant', cval=0.0, prefilter=True) #apply rotation
            motion_contrast_imgs[echo_num].append(motion_img)
            
            
    return np.asarray(motion_contrast_imgs)   


def contrast_and_motion(angles, TEs, T2, PD, mask):
    #return:
    # array of shape [Echos, shots, H, W] or [Echos, len(angles), H, W]
    Echo_imgs = Echo_image_gen(T2=T2, PD=PD, mask=mask, TEs=TEs)
    
    Echo_and_shot = motion_gen(Echo_imgs, angles)
    
    return Echo_and_shot

def motion_corrupt_ksp(image,ordering_type):
    
    ordering = np.zeros((image.shape[-1], 2))
    if ordering_type =='alternating':
        Echo_ind = 0
        for i in range(image.shape[-1]):
            ordering[i,0] = np.mod(i,image.shape[1]) #adjascent lines are seperate only by shot number
        for i in range(image.shape[-1]):
            Echo_ind = np.floor(i/image.shape[-1]) #change the echo number being collected after collecting # shot lines
            ordering[i,1] = Echo_ind
            
    corrupt_ksp = np.zeros((image.shape[-2], image.shape[-1])).astype(complex)
    ksp_full = sp.fft(image, axes=(-2,-1))
    for i in range(corrupt_ksp.shape[-1]):
        Echo_num = int(ordering[i,1])
        Shot_num = int(ordering[i,0])
        inter = ksp_full[Echo_num, Shot_num, :,i]
        corrupt_ksp[:,i] = ksp_full[Echo_num, Shot_num, :,i]
    
    corrupt_img = sp.ifft(corrupt_ksp, axes = (-2,-1))
    return corrupt_ksp, corrupt_img


if __name__ == '__main__':
    # loading the files from a particular folder
    files = glob.glob("MDME_DL_datacreation/mdme_estimated_parameter_maps/*.pt")
    args = parser.parse_args()
    ETL = args.etl
    TE_effective = args.TE_effective
    TE_bw_echoes = args.TE_bw_echoes
    Number_TRs = int(288/ETL)
    subject = args.subject # subject index
    slice_index = args.slice # slice index that will be used for motion generation

    for file in files:
        if(file.find('subject{}'.format(subject)) > 0):
            if(file.find('slice{}'.format(slice_index)) > 0):
                filepath = file
            
    parameter_maps = torch.load(filepath)          
    T1_map = parameter_maps['T1_map']
    T2_map = parameter_maps['T2_map']
    PD_map = parameter_maps['PD_map']
    mask   = parameter_maps['mask']

    # creating the TE-vals array, different cases for odd and even case
    if (ETL%2==1):
        TE_vals = TE_effective + TE_bw_echoes*np.linspace(-int(ETL/2),int(ETL/2),num=ETL)
    elif(ETL%2==0):
        TE_vals = TE_effective + TE_bw_echoes*np.linspace(-(ETL/2-0.5),(ETL/2-0.5),num=ETL)

    # generating the motion artifacts
    # FSE images
    thetas = np.random.normal(loc=0.0, scale=1.0, size=18)
    all_imgs = contrast_and_motion(angles=thetas, TEs=TE_vals, T2=T2_map, PD=PD_map, mask=mask)
    m_ksp, m_img = motion_corrupt_ksp(image = all_imgs, ordering_type = 'alternating')
    FSE_img = m_img
    # GT images
    all_imgs = contrast_and_motion(angles=[0], TEs=[TE_effective], T2=T2_map, PD=PD_map, mask=mask)
    m_ksp, m_img = motion_corrupt_ksp(image = all_imgs, ordering_type = 'alternating')
    GT_img = m_img
    # SE images
    TE_vals = np.ones(ETL)*TE_effective
    all_imgs = contrast_and_motion(angles=thetas, TEs=TE_vals, T2=T2_map, PD=PD_map, mask=mask)
    m_ksp, m_img = motion_corrupt_ksp(image = all_imgs, ordering_type = 'alternating')
    SE_img = m_img

    data_dir = os.getcwd() + '/motion_correction/data_ETL_%s'%(ETL)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir,exist_ok=True)
    torch.save({
        'GT_img': GT_img,
        'SE_img': SE_img,
        'FSE_img':FSE_img,
        'ETL':ETL,
        'TE_effective':TE_effective,
        'TE_bw_echoes':TE_bw_echoes,
        'Number_TRs':Number_TRs,
        'TE_vals':TE_vals,
        },data_dir + '/subject_{}_slice_{}_images.pt'.format(subject,slice_index))
