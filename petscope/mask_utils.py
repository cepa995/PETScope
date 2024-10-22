import os
import functools
import numpy as np
import nibabel as nib
from rich import print
from petscope.constants import REFERENCE_REGIONS

def get_reference_mask(
        template_path: str,
        template_name: str,
        reference_name: str,
        mask_path: str
) -> nib.Nifti1Image:
    """
    Creates 3D, reference mask from a given template based
    on a list of labels which were passed

    :param template_path - absolute path to the template mask
    :param template_name - string which represents name of a template
     (e.g. FreeSurfer)
    :param reference_name - string which represents name of a desired
     reference region (e.g. WholeCerebellum)
    :param mask_path - absolute path where resulting mask will
     be stored
    :returns refernce region/mask as a nibabel object
    """
    reference_region_labels = REFERENCE_REGIONS[template_name][reference_name]
