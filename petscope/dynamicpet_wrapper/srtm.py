import os
import nibabel as nib
import numpy as np
import subprocess
from rich import print
from petscope.dynamicpet_wrapper.utils import generate_subject_json
from petscope.utils import generate_docker_run_cmd, copy_file_to_directory, \
    get_reference_region_mask, get_target_region_mask, c3d_binarize_image, \
    c3d_compute_statistics
from petscope.constants import PET_DEP_DOCKER_IMAGE

def compute_target_region_stats(
        dvr_path: str,
        template_path: str,
        template_name: str,
        target_region: str,
        output_dir: str
) -> None:
        # Create a target region mask from a specified template
        target_mask_path = os.path.join(output_dir, f"{target_region}.nii.gz")
        _ = get_target_region_mask(
            template_path=template_path,
            template_name=template_name,
            target_name=target_region,
            mask_out=target_mask_path
        )

        # Binarize the target region
        target_mask_binary_path = os.path.join(output_dir, f'{target_region}_binary.nii.gz')
        c3d_binarize_image(
             target_mask_path,
             target_mask_binary_path
        )

        # Compute basic DVR image statistics
        stats_file_path = os.path.join(output_dir, "statistics.csv")
        c3d_compute_statistics(
             dvr_path,
             target_mask_binary_path,
             stats_file_path
        )


def call_srtm(
    pet_4d_path: str,
    template_path: str,
    template_name: str,
    reference_region: str,
    target_region: str,
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
        template_path (str): Absolute path to the template from which reference and target regions
                            will be extracted
        template_name (str): Name of the template being used
        reference_region (str): Name of the reference region
        target_region (str): Name of the target region
        output_dir (str): Absolute path to the directory where results will be stored.
        model (str, optional): The kinetic model to use (default: 'SRTMZhou2003').
        fwhm (str, optional): Full Width at Half Maximum describing the spatial resolution 
                              of the imaging system (default: '5').

    Raises:
        SRTMDynamicPETException: If the SRTM computation fails.

    Example:
        call_srtm(
            pet_4d_path="/path/to/pet_image.nii",
            template_path="/path/to/template.nii",
            reference_region="WholeCerebellum",
            target_region="Hippocampus",
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

    # Create a reference region mask from a specified template
    reference_mask_path = os.path.join(output_dir, "reference_region.nii.gz")
    _ = get_reference_region_mask(
        template_path=template_path,
        template_name=template_name,
        reference_name=reference_region,
        mask_out=reference_mask_path
    )

    # Copy files to the mounted directory
    _ = copy_file_to_directory(subject_pet_json, output_dir)
    pet_4d_mounted_path = copy_file_to_directory(pet_4d_path, output_dir)

    # Construct the command line for the SRTM tool
    command_line = [
        "kineticmodel", pet_4d_mounted_path,
        "--model", model,
        "--refmask", reference_mask_path,
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

    try:
        # Run the Docker command
        subprocess.run(docker_cmd, shell=False, check=True)

        # Compute statistics over the desired TARGET region        
        dvr_path = [os.path.join(srtm_results_mounted_dir, f) for f in os.listdir(srtm_results_mounted_dir) if f.endswith('SRTM_meas-dvr_mimap.nii')][0]
        compute_target_region_stats(
             dvr_path=dvr_path,
             template_path=template_path,
             template_name=template_name,
             target_region=target_region,
             output_dir=output_dir
        )
        print("\t:white_heavy_check_mark: [bold green]SRTM was Successful!")
    except Exception as e:
        from petscope.exceptions import SRTMDynamicPETException
        raise SRTMDynamicPETException(f"Could not compute SRTM for {pet_4d_path}") from e
