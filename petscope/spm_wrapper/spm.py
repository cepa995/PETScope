import os
import shutil
import subprocess
from rich import print
from typing import Dict
from petscope.utils import generate_docker_run_cmd
from petscope.spm_wrapper.spm_scripts import get_realignment_script, prepare_spm_4d_data
from petscope.constants import SPM_DOCKER_IMAGE

# Constants for result keys
PET_MEAN = "mean"  # Key for the mean PET volume
PET_REALIGN = "realign"  # Key for the realigned PET volume
PET_TRANSFORM = "transform"  # Key for the transformation matrix
RP_TXT = "rp_txt"  # Key for the realignment parameter text file

def spm_realignment(
        realignment_out_dir: str,
        pet_4d_path: str
) -> Dict[str, str]:
    """
    Wrapper for SPM Realignment using a Dockerized MATLAB/SPM runtime.

    This function handles the realignment of a 4D PET image using SPM's realignment functionality.
    It prepares the input data, generates the necessary SPM script, and executes the MATLAB runtime
    in a Docker container to perform the realignment.

    Args:
        realignment_out_dir (str): Absolute path to the directory where realignment results will be stored.
        pet_4d_path (str): Absolute path to the 4D PET image to be realigned.

    Returns:
        Dict[str, str]: A dictionary containing the paths to the realignment results:
            - `PET_REALIGN`: Path to the realigned 4D PET image.
            - `PET_MEAN`: Path to the mean PET volume image.
            - `PET_TRANSFORM`: Path to the transformation matrix file (.mat).
            - `RP_TXT`: Path to the realignment parameters text file (.txt).

    Raises:
        SPMRealignmentException: If the SPM realignment process fails.

    Example:
        realignment_results = spm_realignment(
            realignment_out_dir="/path/to/output",
            pet_4d_path="/path/to/pet_4d_image.nii"
        )
        print(realignment_results)
    """
    # Ensure the output directory exists and copy the input PET image
    os.makedirs(realignment_out_dir, exist_ok=True)
    shutil.copy(pet_4d_path, realignment_out_dir)

    # Prepare a list of 3D volumes from the 4D PET image for SPM
    list_of_3d_volumes = prepare_spm_4d_data(
        file_path=os.path.join(realignment_out_dir, os.path.basename(pet_4d_path))
    )

    # Generate the MATLAB script for SPM realignment
    script_path = get_realignment_script(
        pet3d_volumes=list_of_3d_volumes,
        output_dir=realignment_out_dir,
        script_name='realignment.m'
    )

    # Generate the Docker command for running the MATLAB/SPM runtime
    docker_cmd = generate_docker_run_cmd(
        image_name=SPM_DOCKER_IMAGE,
        mount_points={realignment_out_dir: realignment_out_dir},
        remove_container=True,
        commands=["script", script_path]
    )

    # Execute the Docker command to perform realignment
    try:
        subprocess.run(docker_cmd)
        print("\t:white_heavy_check_mark: [bold green]SPM Realignment was Successful!")
    except Exception:
        from petscope.exceptions import SPMRealignmentException
        raise SPMRealignmentException(f"Could not perform SPM Realignment for {pet_4d_path}")
    
    # Collect and return realignment results
    results = os.listdir(realignment_out_dir)
    return {
        PET_REALIGN: [os.path.join(realignment_out_dir, f) 
                      for f in results if f.startswith(f"r{os.path.basename(pet_4d_path)}")][0],
        PET_MEAN: [os.path.join(realignment_out_dir, f) 
                   for f in results if f.startswith("mean")][0],
        PET_TRANSFORM: [os.path.join(realignment_out_dir, f) 
                        for f in results if f.endswith(".mat")][0],
        RP_TXT: [os.path.join(realignment_out_dir, f) 
                 for f in results if f.endswith(".txt")][0],
    }
