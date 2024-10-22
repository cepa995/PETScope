import os
import nibabel as nib
import subprocess
import numpy as np
from nilearn.image import mean_img, concat_imgs
from scripts.constants import C3D_BINARY

def change_dtype(image_path, output_path, dtype):
    """
    Change datatype of an image

    :param image_path - absolute path to the image which needs orientation change
    :param output_path - absolute path to the resulting image
    :param orientation_code - e.g., char, uchar, short, ushort, int, uint
    """
    subprocess.run([
        C3D_BINARY,
        image_path,
        '-type', dtype,
        "-o", output_path
    ])

def change_orientation(image_path, output_path, orientation_code='RSA'):
    """
    Change orientatiion of an image according to the given code (default RSA)

    :param image_path - absolute path to the image which needs orientation change
    :param output_path - absolute path to the resulting image
    :param orientation_code - e.g., RSA, LPI, RAI
    """
    subprocess.run([
        C3D_BINARY,
        image_path,
        '-swapdim', orientation_code,
        "-o", output_path
    ])

def compute_4d_image(volume_dir, img_4d_out):
    """
    Creates a 4D Image from 3D Volume dir

    :param volume_dir - absolute patht to 3D volume dir
    :param img_4d_out - absolute path to 4D volume image
    """
    volumes_nii = [nib.load(os.path.join(volume_dir, f)) for f in os.listdir(volume_dir)]
    img_4d_nii  = concat_imgs(volumes_nii)
    nib.save(img_4d_nii, img_4d_out)
    return img_4d_nii

def compute_mean_volume(volume_dir, mean_3d_out):
    """
    Creates mean 3D volume from list of 3D volumes

    :param volume_dir - absolute patht to 3D volume dir
    :param mean_3d_vol - absolute path to 3D volume path
    """
    volumes_nii = [nib.load(os.path.join(volume_dir, f)) for f in os.listdir(volume_dir)]
    mean_3d_image  = mean_img(volumes_nii, volumes_nii[0].affine)
    nib.save(mean_3d_image, mean_3d_out)
    return mean_3d_image

def compute_3D_volume(nifti_4d_path, output_file):
    """
    Converts 4D Image into 3D Volumes

    :param nifti_4d_path - absolute path to 4D nifti image
    :param output_dir - absolute path to location where 3D
     mean 3D volume will be stored
    """
    nifti_4d      = nib.load(nifti_4d_path)
    nifti_4d_data = nifti_4d.get_fdata()
    # Check if the image has the right amount of dimensions
    assert len(nifti_4d_data.shape) == 4, "{} PET image should have exactly 4 Dimensions".format(nifti_4d_path) 
    # Check if the image is not empty
    assert nifti_4d_data.min() != nifti_4d_data.max(), "{} PET image appears to be empty".format(nifti_4d_path)
    # Convert to 3D volume (Expected Shape X,Y,Z,C)
    rank          = 0.20
    num_frames    = nifti_4d_data.shape[3]
    first_frame   = int(np.floor(num_frames * rank))
    last_frame    = num_frames
    volume_subset = nifti_4d_data[:, :, :, first_frame:last_frame]
    volume_src    = np.mean(volume_subset, axis=3)
    output_3d_img = nib.Nifti1Image(volume_src, nifti_4d.affine)
    nib.save(output_3d_img,output_file)

def convert_4d_to_3d(img_4d_path, img_3d_dir, prefix='pet_3d_', orientation=None):
    """
    Convert 4D image into list of 3D volumes

    :param img_4d_path - absolute path to 4d image
    :param orientation - orientation code (if applicable)
    """
    os.makedirs(img_3d_dir, exist_ok=True)
    img_4d_nii  = nib.load(img_4d_path)
    img_4d_data = img_4d_nii.get_fdata()
    num_3d_imgs = img_4d_data.shape[-1]
    for idx in range(0, num_3d_imgs):
        img_3d_vol_data = img_4d_data[..., idx]
        img_3d_vol_nii  = nib.Nifti1Image(img_3d_vol_data, img_4d_nii.affine)
        img_3d_vol_out  = os.path.join(img_3d_dir, '{}_{}.nii'.format(prefix, idx))
        nib.save(img_3d_vol_nii, img_3d_vol_out)
        if orientation != None:
            #NOTE: We cannot change PET 4D to RSA, we must split it 1st into sequence of 3D volumes
            # and then change each to RSA (C3D works with 3D images not 4D) 
            change_orientation(
                image_path=img_3d_vol_out,
                orientation_code=orientation,
                output_path=img_3d_vol_out
            )
