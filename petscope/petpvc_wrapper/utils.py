import os
import functools
import numpy as np
import nibabel as nib
from rich import print
from petscope.constants import REFERENCE_REGIONS, PVC_SUPPORTED_METHODS

def check_if_pvc_method_is_supported(method):
    """True if PVC method is indeed supported, False otherwise"""
    return True if method in PVC_SUPPORTED_METHODS else False

def petpvc_create_4d_mask(
        template_path: str,
        template_name: str,
        reference_name: str,
        mask_4d_out: str
    ) -> nib.Nifti1Image:
    """
    Create a 4D mask from a 3D mask and a list of desired labels.
    (This function is PETPVC specific.)

    :param template_path - absolute path to the template mask
    :param template_name - string which represents name of a template
     (e.g. FreeSurfer)
    :param reference_name - string which represents name of a desired
     reference region (e.g. WholeCerebellum)
    :param mask_4d_out: absolute path where the resulting 4D mask will be 
     saved
    :returns: 4D Nifti1Image object
    """
    reference_region_labels = REFERENCE_REGIONS[template_name][reference_name]
    # Create output directory if it doesn't exist
    dirname = os.path.dirname(mask_4d_out)
    os.makedirs(dirname, exist_ok=True)

    # Load the 3D mask data
    mask_3d_nii = nib.load(template_path)
    mask_3d_data = mask_3d_nii.get_fdata().astype(np.uint16)

    # Inverted background - required for PETPVC as the sum of all pixels across
    # the 4th dimension has to be equal to 1.
    print("\t[bold blue][CREATE VOLUME]: [green]Inverted Labels Image")
    
    # Create a binary mask that combines all labels
    mask = functools.reduce(np.logical_or, (mask_3d_data == lbl for lbl in reference_region_labels))
    inverted_image = np.where(mask, mask_3d_data, 0)
    inverted_image[inverted_image != 0] = 1
    inverted_image = 1 - inverted_image
    
    # Save inverted labels as a Nifti image
    inverted_image_nii = nib.Nifti1Image(inverted_image.astype(np.uint16), mask_3d_nii.affine)
    inverted_image_path = os.path.join(dirname, "inverted_labels.nii.gz")
    nib.save(inverted_image_nii, inverted_image_path)

    # Generate list of 3D volumes based on the given labels
    mask_3d_lbl_volumes = [inverted_image_path]
    for lbl in reference_region_labels:
        print(f"\t[bold blue][CREATE VOLUME] [green]For label[/] {lbl}")
        mask_3d_lbl_volume = mask_3d_data.copy().astype(np.uint16)
        mask_3d_lbl_volume[mask_3d_lbl_volume != lbl] = 0
        mask_3d_lbl_volume[mask_3d_lbl_volume == lbl] = 1
        
        # Save volume for visualization/debugging purposes
        mask_3d_lbl_volume_nii = nib.Nifti1Image(mask_3d_lbl_volume.astype(np.uint16), mask_3d_nii.affine)
        mask_3d_lbl_volume_path = os.path.join(dirname, f"mask_3d_volume_{lbl}.nii.gz")
        nib.save(mask_3d_lbl_volume_nii, mask_3d_lbl_volume_path)
        
        mask_3d_lbl_volumes.append(mask_3d_lbl_volume_path)
    # Create and save the 4D mask required for Partial Volume Correction (PVC)
    print("\t:white_heavy_check_mark: [bold green]SUCCESS! ")
    mask_4d_nii = nib.concat_images(mask_3d_lbl_volumes)
    nib.save(mask_4d_nii, mask_4d_out)

    return mask_4d_nii
