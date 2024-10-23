import os
import functools
import numpy as np
import nibabel as nib

from nilearn.image import concat_imgs
from src.logger import logger


def petpvc_create_4d_mask(mask_3d_path, list_of_labels, mask_4d_out):
    """
    Create a 4D mask from a 3D mask and a list of desired labels.
    (This function is PETPVC specific.)

    :param mask_3d_path: absolute path to the 3D mask to be converted
    :param list_of_labels: list of labels, each representing a single volume in the 4D mask
    :param mask_4d_out: absolute path where the resulting 4D mask will be saved
    :returns: 4D Nifti1Image object
    """
    # Create output directory if it doesn't exist
    dirname = os.path.dirname(mask_4d_out)
    os.makedirs(dirname, exist_ok=True)

    # Load the 3D mask data
    mask_3d_nii = nib.load(mask_3d_path)
    mask_3d_data = mask_3d_nii.get_fdata()

    # Inverted background - required for PETPVC as the sum of all pixels across
    # the 4th dimension has to be equal to 1.
    logger.info("Generating 1st volume - Inverted Labels Image")
    
    # Create a binary mask that combines all labels
    mask = functools.reduce(np.logical_or, (mask_3d_data == lbl for lbl in list_of_labels))
    inverted_image = np.where(mask, mask_3d_data, 0)
    inverted_image[inverted_image != 0] = 1
    inverted_image = 1 - inverted_image.astype(np.uint8)
    
    # Save inverted labels as a Nifti image
    inverted_image_nii = nib.Nifti1Image(inverted_image.astype(np.uint8), mask_3d_nii.affine)
    inverted_image_path = os.path.join(dirname, "inverted_labels.nii.gz")
    nib.save(inverted_image_nii, inverted_image_path)

    # Generate list of 3D volumes based on the given labels
    mask_3d_lbl_volumes = [inverted_image_nii]
    
    for lbl in list_of_labels:
        logger.info(f"Generating Volume for Label {lbl}")
        mask_3d_lbl_volume = mask_3d_data.copy()
        mask_3d_lbl_volume[mask_3d_lbl_volume != lbl] = 0
        mask_3d_lbl_volume[mask_3d_lbl_volume == lbl] = 1
        
        # Save volume for visualization/debugging purposes
        mask_3d_lbl_volume_nii = nib.Nifti1Image(mask_3d_lbl_volume.astype(np.uint8), mask_3d_nii.affine)
        mask_3d_lbl_volume_path = os.path.join(dirname, f"mask_3d_volume_{lbl}.nii.gz")
        nib.save(mask_3d_lbl_volume_nii, mask_3d_lbl_volume_path)
        
        mask_3d_lbl_volumes.append(mask_3d_lbl_volume_nii)

    # Create and save the 4D mask required for Partial Volume Correction (PVC)
    logger.info("Generating 4D Mask required for Partial Volume Correction (PVC)")
    mask_4d_nii = concat_imgs(mask_3d_lbl_volumes)
    nib.save(mask_4d_nii, mask_4d_out)

    return mask_4d_nii
