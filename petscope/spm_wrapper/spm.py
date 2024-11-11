import os
import shutil
import subprocess
from rich import print
from typing import Dict
from petscope.utils import generate_docker_run_cmd
from petscope.spm_wrapper.spm_scripts import get_realignment_script, prepare_spm_4d_data
from petscope.constants import SPM_DOCKER_IMAGE

PET_MEAN = "mean"
PET_REALIGN = "realign"
PET_TRANSFORM = "transform"
RP_TXT = "rp_txt"

def spm_realignment(
        realignment_out_dir: str,
        pet_4d_path: str
) -> Dict[str, str]:
    """
    SPM Realignment wrapper

    :param realignment_out_dir: absolute path to the directory
        where realignment results will be stored
    :param pet_4d_path: absolute path to PET 4D image
    :returns: dictionary with realignment results
    """
    # Move all relevant data to a directory that is going to be mounted inside of
    # a Docker container
    os.makedirs(realignment_out_dir, exist_ok=True)
    shutil.copy(pet_4d_path, realignment_out_dir)

    # Generate string which represents list of 3D volumes that are used to construct
    # PET 4D Image - this format is required by SPM
    list_of_3d_volumes = prepare_spm_4d_data(
        file_path=os.path.join(realignment_out_dir, os.path.basename(pet_4d_path))
    )

    # Generate a realignment.m script which should be run by MATLAB runtime
    script_path = get_realignment_script(
        pet3d_volumes=list_of_3d_volumes,
        output_dir=realignment_out_dir,
        script_name='realignment.m'
    )

    # Generate a command which will be run by MATLAB/SPM runtime Docker container
    docker_cmd = generate_docker_run_cmd(
        image_name=SPM_DOCKER_IMAGE,
        mount_points={realignment_out_dir: realignment_out_dir},
        remove_container=True,
        commands=["script", script_path]
    )

    # Execute the command
    try:
        subprocess.run(docker_cmd)
        print("\t:white_heavy_check_mark: [bold green]SPM Realignment was Successful!")
    except Exception:
        from petscope.exceptions import SPMRealignmentException
        raise SPMRealignmentException(f"Could not perform SPM Realignment for {pet_4d_path}")
    
    # Organize SPM realignment results
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
