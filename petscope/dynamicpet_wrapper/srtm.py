import os
import subprocess
from rich import print
from petscope.dynamicpet_wrapper.utils import generate_subject_json
from petscope.utils import generate_docker_run_cmd, copy_file_to_directory
from petscope.constants import PET_DEP_DOCKER_IMAGE

def call_srtm(pet_4d_path, reference_mask_path, output_dir, model='SRTMZhou2003', fwhm='5'):
    """
    Dynamic PET wrapper for Simplified Reference Tissue Model (SRTM)

    :param pet_4d_path: Absolute path to 4D PET image.
    :param reference_mask_path: Absolute path to reference mask image.
    :param output_dir: Absolute path to the output directory.
    :param model: Model being passed to kineticmodel binary (default: 'SRTMZhou2003').
    :param fwhm: Full Width at Half Maximum describing the spatial resolution of the imaging system (default: '5').
    """

    # Create directory which will be mounted to docker container
    srtm_results_mounted_dir = os.path.join(output_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Generate PET JSON file, if it doesn't exist
    subject_pet_json = generate_subject_json(pet_4d_path)

    # Copy files to the mounted directory using the helper function
    _ = copy_file_to_directory(subject_pet_json, output_dir)
    pet_4d_mounted_path = copy_file_to_directory(pet_4d_path, output_dir)
    reference_mask_mounted_path = copy_file_to_directory(reference_mask_path, output_dir)

    # Construct Dynamic PET command line
    command_line = [
        "kineticmodel", pet_4d_mounted_path,
        "--model", model,
        "--refmask", reference_mask_mounted_path,
        "--outputdir", srtm_results_mounted_dir,
        "--fwhm", fwhm
    ]

    # Generate a command which will be run by MATLAB/SPM runtime Docker container
    docker_cmd = generate_docker_run_cmd(
        image_name=PET_DEP_DOCKER_IMAGE,
        mount_points={output_dir: output_dir},
        remove_container=True,
        commands=command_line
    )
    
    # Run the command
    try:
        subprocess.run(docker_cmd, shell=False, check=True)
        print("\t:white_heavy_check_mark: [bold green]SRTM was Successful!")
    except Exception:
        from petscope.exceptions import SRTMDynamicPETException
        raise SRTMDynamicPETException(f"Could not compute SRTM for {pet_4d_path}")