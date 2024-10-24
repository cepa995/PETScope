import os
import nibabel as nib
import subprocess
import numpy as np
from nilearn.image import mean_img, concat_imgs


def change_dtype(image_path, output_path, dtype):
    """
    Change datatype of an image.

    :param image_path: absolute path to the image which needs datatype change
    :param output_path: absolute path to the resulting image
    :param dtype: target datatype (e.g., char, uchar, short, ushort, int, uint)
    """
    subprocess.run([
        'c3d',
        image_path,
        '-type', dtype,
        "-o", output_path
    ])


def change_orientation(image_path, output_path, orientation_code='RSA'):
    """
    Change orientation of an image according to the given code (default: RSA).

    :param image_path: absolute path to the image which needs orientation change
    :param output_path: absolute path to the resulting image
    :param orientation_code: orientation code (e.g., RSA, LPI, RAI)
    """
    subprocess.run([
        'c3d',
        image_path,
        '-swapdim', orientation_code,
        "-o", output_path
    ])


def compute_4d_image(volume_dir, img_4d_out):
    """
    Creates a 4D Image from a directory of 3D Volumes.

    :param volume_dir: absolute path to 3D volume directory
    :param img_4d_out: absolute path to output 4D volume image
    :returns: 4D Nifti1Image object
    """
    volumes_nii = [nib.load(os.path.join(volume_dir, f)) for f in os.listdir(volume_dir)]
    img_4d_nii = concat_imgs(volumes_nii)
    nib.save(img_4d_nii, img_4d_out)
    return img_4d_nii


def compute_mean_volume(volume_dir, mean_3d_out):
    """
    Creates a mean 3D volume from a list of 3D volumes.

    :param volume_dir: absolute path to 3D volume directory
    :param mean_3d_out: absolute path to output mean 3D volume
    :returns: mean 3D Nifti1Image object
    """
    volumes_nii = [nib.load(os.path.join(volume_dir, f)) for f in os.listdir(volume_dir)]
    mean_3d_image = mean_img(volumes_nii, volumes_nii[0].affine)
    nib.save(mean_3d_image, mean_3d_out)
    return mean_3d_image


def compute_3D_volume(nifti_4d_path, output_file):
    """
    Converts a 4D Image into a 3D Volume.

    :param nifti_4d_path: absolute path to 4D nifti image
    :param output_file: absolute path to the output 3D volume
    :returns: 3D Nifti1Image object
    """
    nifti_4d = nib.load(nifti_4d_path)
    nifti_4d_data = nifti_4d.get_fdata()
    
    # Check if the image has 4 dimensions
    assert len(nifti_4d_data.shape) == 4, f"{nifti_4d_path} image must have exactly 4 dimensions"
    
    # Check if the image is not empty
    assert nifti_4d_data.min() != nifti_4d_data.max(), f"{nifti_4d_path} image appears to be empty"
    
    # Convert to 3D volume (Expected Shape X, Y, Z, C)
    rank = 0.20
    num_frames = nifti_4d_data.shape[3]
    first_frame = int(np.floor(num_frames * rank))
    last_frame = num_frames
    volume_subset = nifti_4d_data[:, :, :, first_frame:last_frame]
    volume_src = np.mean(volume_subset, axis=3)
    
    output_3d_img = nib.Nifti1Image(volume_src, nifti_4d.affine)
    nib.save(output_3d_img, output_file)
    return output_3d_img


def convert_4d_to_3d(img_4d_path, img_3d_dir, prefix='pet_3d_', orientation=None):
    """
    Convert a 4D image into a sequence of 3D volumes.

    :param img_4d_path: absolute path to 4D image
    :param img_3d_dir: directory to store the 3D volumes
    :param prefix: prefix for 3D volume filenames
    :param orientation: orientation code (if applicable)
    """
    os.makedirs(img_3d_dir, exist_ok=True)
    img_4d_nii = nib.load(img_4d_path)
    img_4d_data = img_4d_nii.get_fdata()
    num_3d_imgs = img_4d_data.shape[-1]
    
    for idx in range(num_3d_imgs):
        img_3d_vol_data = img_4d_data[..., idx]
        img_3d_vol_nii = nib.Nifti1Image(img_3d_vol_data, img_4d_nii.affine)
        img_3d_vol_out = os.path.join(img_3d_dir, f'{prefix}_{idx}.nii')
        nib.save(img_3d_vol_nii, img_3d_vol_out)
        
        if orientation is not None:
            # Split 4D image into sequence of 3D volumes and change orientation for each
            change_orientation(
                image_path=img_3d_vol_out,
                orientation_code=orientation,
                output_path=img_3d_vol_out
            )
