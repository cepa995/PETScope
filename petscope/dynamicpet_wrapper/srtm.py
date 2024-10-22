import os
import subprocess
from petscope.dynamicpet_wrapper.utils import generate_subject_json

def call_srtm(pet_4d_path, reference_mask_path, output_dir, model='SRTMZhou2003', fwhm='5'):
    """
    Dynamic PET rapper for Simplified Reference Tissue Model (SRTM)

    :param pet_4d_path - absolute path to 4D PET image
    :param reference_mask_path - absolute path to reference mask image
    :param output_dir - absolute path to the output directory
    :param model - model being passed to kineticmodel binary
    :param fwhm - Ful Width at Half Maximum used to describe the spatial resolution of the
     imageing system
    """
    # Generate PET JSON file, if it doesn't exist
    generate_subject_json(pet_4d_path)
    # Run Simplified Reference Tissue Model (SRTM)
    command_line = [
        "kineticmodel", pet_4d_path,
        "--model", model,
        "--refmask", reference_mask_path,
        "--outputdir", output_dir,
        "--fwhm", fwhm
    ]
    try:
        subprocess.run(command_line)
    except Exception as err:
        raise Exception(err)