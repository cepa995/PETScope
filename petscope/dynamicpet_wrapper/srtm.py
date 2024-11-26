import os
import subprocess
from rich import print
from petscope.dynamicpet_wrapper.utils import generate_subject_json
from petscope.utils import generate_docker_run_cmd, copy_file_to_directory
from petscope.constants import PET_DEP_DOCKER_IMAGE

def call_srtm(
    pet_4d_path: str,
    reference_mask_path: str,
    output_dir: str,
    model: str = 'SRTMZhou2003',
    fwhm: str = '5'
) -> None:
    """
    Executes the Simplified Reference Tissue Model (SRTM) for Dynamic PET analysis.

    This function serves as a wrapper for running SRTM using a Docker container that encapsulates
    the required dependencies. It generates the necessary inputs, constructs the command,
    and executes the SRTM pipeline.

    Args:
        pet_4d_path (str): Absolute path to the 4D PET image.
        reference_mask_path (str): Absolute path to the reference region mask image.
        output_dir (str): Absolute path to the directory where results will be stored.
        model (str, optional): The kinetic model to use (default: 'SRTMZhou2003').
        fwhm (str, optional): Full Width at Half Maximum describing the spatial resolution 
                              of the imaging system (default: '5').

    Raises:
        SRTMDynamicPETException: If the SRTM computation fails.

    Example:
        call_srtm(
            pet_4d_path="/path/to/pet_image.nii",
            reference_mask_path="/path/to/reference_mask.nii",
            output_dir="/path/to/output",
            model="SRTMZhou2003",
            fwhm="5"
        )
    """
    # Create a directory for results that will be mounted to the Docker container
    srtm_results_mounted_dir = os.path.join(output_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Generate the subject's PET JSON file if it doesn't already exist
    subject_pet_json = generate_subject_json(pet_4d_path)

    # Copy files to the mounted directory
    _ = copy_file_to_directory(subject_pet_json, output_dir)
    pet_4d_mounted_path = copy_file_to_directory(pet_4d_path, output_dir)
    reference_mask_mounted_path = copy_file_to_directory(reference_mask_path, output_dir)

    # Construct the command line for the SRTM tool
    command_line = [
        "kineticmodel", pet_4d_mounted_path,
        "--model", model,
        "--refmask", reference_mask_mounted_path,
        "--outputdir", srtm_results_mounted_dir,
        "--fwhm", fwhm
    ]

    # Generate the Docker command for running the SRTM pipeline
    docker_cmd = generate_docker_run_cmd(
        image_name=PET_DEP_DOCKER_IMAGE,
        mount_points={output_dir: output_dir},
        remove_container=True,
        commands=command_line
    )

    # Run the Docker command
    try:
        subprocess.run(docker_cmd, shell=False, check=True)
        print("\t:white_heavy_check_mark: [bold green]SRTM was Successful!")
    except Exception as e:
        from petscope.exceptions import SRTMDynamicPETException
        raise SRTMDynamicPETException(f"Could not compute SRTM for {pet_4d_path}") from e
