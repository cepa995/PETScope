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
    Executes Partial Volume Correction (PVC) using the Iterative Yang method.

    This function uses the `PETPVC` package from UCL to perform PVC on a given 3D PET volume
    using the Iterative Yang method. It requires a 4D mask and the Point Spread Function (PSF) 
    resolution parameters along the x, y, and z axes.

    Args:
        pet_3d_volume_path (str): Absolute path to the input 3D PET volume file.
        pet_4d_mask_path (str): Absolute path to the 4D mask file, constructed as per 
                                the PETPVC documentation.
        pet_3d_volume_pvc_path (str): Absolute path where the PVC-corrected 3D volume will be saved.
        x (str, optional): PSF resolution along the x-axis in millimeters. Default is "6.0".
        y (str, optional): PSF resolution along the y-axis in millimeters. Default is "6.0".
        z (str, optional): PSF resolution along the z-axis in millimeters. Default is "6.0".

    Returns:
        None

    Raises:
        Exception: If the PETPVC process fails to execute.

    Example:
        run_petpvc_iterative_yang(
            pet_3d_volume_path="/path/to/3d_volume.nii",
            pet_4d_mask_path="/path/to/4d_mask.nii",
            pet_3d_volume_pvc_path="/path/to/output_pvc_volume.nii",
            x="5.0",
            y="5.0",
            z="5.0"
        )

    Notes:
        - The Iterative Yang method is part of the PETPVC software package.
        - For more details, see the PETPVC documentation: https://github.com/UCL/PETPVC
    """
    # Command to execute PETPVC with the Iterative Yang method
    command_petpvc_iterative_yang = [
        'petpvc', "-i", pet_3d_volume_path,
        "-m", pet_4d_mask_path, "-o", pet_3d_volume_pvc_path,
        "--pvc", "IY", "-x", x, "-y", y, "-z", z
    ]
    try:
        # Run the PETPVC command
        subprocess.run(command_petpvc_iterative_yang, check=True)
    except Exception as err:
        raise Exception(f"Error executing PETPVC Iterative Yang method: {err}")
    
    # Print success message
    print("\t:white_heavy_check_mark: [bold green]SUCCESS! ")
