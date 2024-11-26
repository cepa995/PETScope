import os
import functools
import numpy as np
import nibabel as nib
from rich import print
from petscope.constants import REFERENCE_REGIONS, PVC_SUPPORTED_METHODS


def check_if_pvc_method_is_supported(method: str) -> bool:
    """
    Checks if the specified Partial Volume Correction (PVC) method is supported.

    Args:
        method (str): The name of the PVC method (e.g., "IterativeYang").

    Returns:
        bool: True if the PVC method is supported, False otherwise.
    """
    return method in PVC_SUPPORTED_METHODS

def petpvc_create_4d_mask(
        template_path: str,
        template_name: str,
        reference_name: str,
        mask_4d_out: str,
        debug: bool = False
    ) -> nib.Nifti1Image:
    """
    Creates a 4D mask from a 3D mask and a list of desired labels for Partial Volume Correction (PVC).

    This function is specific to PETPVC and generates a 4D mask required for PVC. 
    It combines reference region labels into individual 3D volumes and concatenates them 
    into a 4D NIfTI image.

    Args:
        template_path (str): Absolute path to the 3D template mask.
        template_name (str): Name of the template (e.g., "FreeSurfer").
        reference_name (str): Name of the reference region (e.g., "WholeCerebellum").
        mask_4d_out (str): Absolute path where the resulting 4D mask will be saved.
        debug (bool, optional): If True, saves intermediate 3D volumes for debugging. Defaults to False.

    Returns:
        nib.Nifti1Image: The resulting 4D NIfTI image.

    Example:
        petpvc_create_4d_mask(
            template_path="/path/to/template.nii",
            template_name="FreeSurfer",
            reference_name="WholeCerebellum",
            mask_4d_out="/path/to/output_4d_mask.nii.gz",
            debug=True
        )
    """
    # Get labels for the specified reference region
    reference_region_labels = REFERENCE_REGIONS[template_name][reference_name]

    # Create the output directory if it doesn't exist
    dirname = os.path.dirname(mask_4d_out)
    os.makedirs(dirname, exist_ok=True)

    # Load the 3D mask
    mask_3d_nii = nib.load(template_path)
    mask_3d_data = mask_3d_nii.get_fdata().astype(np.uint16)

    # Create an inverted image for the background
    print("\t[bold blue][CREATE VOLUME]: [green]Inverted Labels Image")
    mask = functools.reduce(np.logical_or, (mask_3d_data == lbl for lbl in reference_region_labels))
    inverted_image = np.where(mask, mask_3d_data, 0)
    inverted_image[inverted_image != 0] = 1
    inverted_image = 1 - inverted_image

    # Save the inverted labels as a NIfTI image
    inverted_image_nii = nib.Nifti1Image(inverted_image.astype(np.uint16), mask_3d_nii.affine)
    inverted_image_path = os.path.join(dirname, "inverted_labels.nii.gz")
    nib.save(inverted_image_nii, inverted_image_path)

    # Generate individual 3D volumes for each label
    mask_3d_lbl_volumes = [inverted_image_path]
    for lbl in reference_region_labels:
        print(f"\t[bold blue][CREATE VOLUME] [green]For label[/] {lbl}")
        mask_3d_lbl_volume = mask_3d_data.copy().astype(np.uint16)
        mask_3d_lbl_volume[mask_3d_lbl_volume != lbl] = 0
        mask_3d_lbl_volume[mask_3d_lbl_volume == lbl] = 1

        # Save the 3D volume for debugging if enabled
        mask_3d_lbl_volume_nii = nib.Nifti1Image(mask_3d_lbl_volume.astype(np.uint16), mask_3d_nii.affine)
        mask_3d_lbl_volume_path = os.path.join(dirname, f"mask_3d_volume_{lbl}.nii.gz")
        if debug:
            nib.save(mask_3d_lbl_volume_nii, mask_3d_lbl_volume_path)

        mask_3d_lbl_volumes.append(mask_3d_lbl_volume_path)

    # Concatenate the 3D volumes into a 4D mask
    print("\t:white_heavy_check_mark: [bold green]SUCCESS! ")
    mask_4d_nii = nib.concat_images(mask_3d_lbl_volumes)
    nib.save(mask_4d_nii, mask_4d_out)

    return mask_4d_nii
