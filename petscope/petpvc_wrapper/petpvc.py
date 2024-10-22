import os
import subprocess
from rich import print

def run_petpvc_iterative_yang(
        pet_3d_volume_path: str,
        pet_4d_mask_path: str,
        pet_3d_volume_pvc_path: str,
        x: str = "6.0",
        y: str = "6.0",
        z: str = "6.0"
) -> None:
    """
    Executes Partial Volume Correction using Iterative Yang method 
    (implemented by UCL/PETVC)

    :param pet_3d_volume_path - absolute path to 3D volume 
    :param pet_4d_mask_path - absolute path to 4D mask which is constructed
     based on the information provided in UCL/PETPVC documentation 
     (https://github.com/cepa995/PETPVC)
    :param pet_3d_volume_pvc_path - absolute path to PVC volume
    :param x Point Spread Function (PSF) resultion in millimeters along x axis
    :param y Point Spread Function (PSF) resultion in millimeters along y axis
    :param z Point Spread Function (PSF) resultion in millimeters along z axis
    """
    # Run PET Partial Volume Correction iterative Yang method (UCL/PETPVC)
    command_petpvc_iterative_yang = [
        'petpvc', "-i", pet_3d_volume_path,
        "-m", pet_4d_mask_path, "-o", pet_3d_volume_pvc_path,
        "--pvc", "IY", "-x", x, "-y", y, "-z", z
    ]
    try:
        subprocess.run(command_petpvc_iterative_yang)
    except Exception as err:
        raise Exception(err)
    print("\t:white_heavy_check_mark: [bold green]SUCCESS! ")