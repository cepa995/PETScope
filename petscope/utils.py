import os
import re
import nibabel as nib
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from nilearn.image import mean_img, concat_imgs
from scipy.signal import savgol_filter

def compute_time_activity_curve(
        pet_image_path: str,
        referemce_region_path: str,
        time_activity_curve_out: str) -> None:
    """
    Computes a Time Activity Curve (TAC) over the given reference
    region (make sure to specify one of the supported reference
    regions).

    :param pet_3d_image_path - absolute path to mean 3D or 4D PET image
    :param reference_region_path - absolute path to a reference region
    :param time_activity_curve_out - absolute path to TAC out
    """
    # Check if image and reference region are indeed in the same space
    if not c3d_space_check(pet_image_path, referemce_region_path):
        from petscope.exceptions import NotSamePhysicalSpaceException
        raise NotSamePhysicalSpaceException(
            f"Template image {pet_image_path} is not in the same space as the
             reference region {referemce_region_path}"
        )
    
    # Load PET image
    pet_img_nii = nib.load(pet_image_path)
    pet_img_data = pet_img_nii.get_fdata()

    # Load Reference Region
    reference_region_img_nii = nib.load(referemce_region_path)
    reference_region_img_data = reference_region_img_nii.get_fdata()

    # Calculate the TAC by averaging over the ROI for each time frame
    tac = []
    for t in range(pet_img_data.shape[3]): # Looping over time-frames
        average_activity = np.mean(pet_img_data[reference_region_img_data, t])
        tac.append(average_activity)
    
    # Convert TAC to numpy array for easier handling
    tac = np.array(tac)

    # Plot the Time Activity Curve and save it
    plt.figure(figsize=(10, 5))
    plt.plot(tac, marker='o')
    plt.title('Time Activity Curve (TAC)')
    plt.xlabel('Time Frame')
    plt.ylabel('Average Activity')
    plt.grid()

    # Save the figure
    plt.savefig(time_activity_curve_out, dpi=300, bbox_inches='tight')  
    plt.close()  

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

def get_orientation(nifti_image_path) -> str:
    """
    Check the NIfTI orientation.
    :param nifti_image_path: absolute path to NIfTI file
    :return: image orientation
    """
    image_nii = nib.load(nifti_image_path)
    x, y, z = nib.aff2axcodes(image_nii.affine)
    return x + y + z

def extract_image_info(image_path):
    """Extracts dimension, bounding box, and orientation info from an image using c3d."""
    c3d_info_cmd = ["c3d", image_path, "-info"]
    result = subprocess.run(c3d_info_cmd, capture_output=True, text=True)
    info = result.stdout

    # Regular expressions to find 'dim', 'bb', and 'orient' values
    dim_match = re.search(r'dim = \[([^\]]+)\]', info)
    bb_match = re.search(r'bb = \{([^\}]+)\}', info)
    orient_match = re.search(r'orient = (\w+)', info)

    # Extract and convert values if matches are found
    dim = [float(x) for x in dim_match.group(1).split(',')] if dim_match else None
    bb = [[float(x) for x in point.replace('[', '').replace(']', '').split()] for point in bb_match.group(1).split('], [')] if bb_match else None
    orient = orient_match.group(1) if orient_match else None

    return dim, bb, orient

def c3d_space_check(image1_path, image2_path) -> bool:
    """
    Utilizes Convert3D tool to get image information,
    parse it, and check if the two images belong to the same space.

    :param image1_path: Absolute path to the 1st image
    :param image2_path: Absolute path to the 2nd image
    :returns: True if images are in the same space, False otherwise
    """
    dim1, bb1, orient1 = extract_image_info(image1_path)
    dim2, bb2, orient2 = extract_image_info(image2_path)

    return dim1 == dim2 and bb1 == bb2 and orient1 == orient2

