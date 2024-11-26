import os
import nibabel as nib
from typing import List

def get_realignment_script(pet3d_volumes: List[str], output_dir: str, script_name: str) -> str:
    """
    Creates a MATLAB script for PET image realignment using SPM.

    This function generates a MATLAB script that configures and runs the realignment process 
    for a series of 3D PET volumes using SPM (Statistical Parametric Mapping).

    Args:
        pet3d_volumes (List[str]): List of absolute paths to 3D PET volume files to be realigned.
        output_dir (str): Directory where the MATLAB script will be saved.
        script_name (str): Name of the MATLAB script file to be created (e.g., `realignment.m`).

    Returns:
        str: Absolute path to the generated MATLAB script.

    Example:
        pet_volumes = ["/path/to/volume1.nii", "/path/to/volume2.nii"]
        script_path = get_realignment_script(
            pet3d_volumes=pet_volumes,
            output_dir="/path/to/output",
            script_name="realignment.m"
        )
        print(f"MATLAB script created at {script_path}")
    """
    # Construct the full path for the script
    script_path = os.path.join(output_dir, script_name)
    
    # Open the file for writing the MATLAB script
    with open(script_path, "w") as fp:
        # Write MATLAB commands for realignment
        fp.writelines(
            [
                # Initialize the SPM job manager
                "spm_jobman('initcfg');\n",
                
                # Add data for realignment
                "matlabbatch{1}.spm.spatial.realign.estwrite.data = {\n",
                "{\n",
            ]
            + pet3d_volumes  # Add the PET 3D volumes
            + [
                "}\n",
                "};\n",
                
                # Realignment estimation options
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.sep = 4;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.rtm = 1;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.interp = 2;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.weight = '';\n",
                
                # Realignment reslicing options
                "matlabbatch{1}.spm.spatial.realign.estwrite.roptions.which = [2 1];\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.roptions.interp = 4;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.roptions.mask = 1;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.roptions.prefix = 'r';\n",
                
                # Run the batch job
                "spm_jobman('run',matlabbatch);\n",
            ]
        )
    return script_path


def prepare_spm_4d_data(file_path: str) -> List[str]:
    """
    Generates a list of 3D PET volumes from a 4D PET image for SPM processing.

    This function mimics the output format of `spm_select` by creating references to 
    individual 3D volumes within a 4D PET image, which is required for SPM realignment.

    Args:
        file_path (str): Absolute path to the 4D PET image (e.g., NIfTI file).

    Returns:
        List[str]: List of strings, each referencing a specific 3D volume in the format
        required by SPM (e.g., `"<file_path>,<index>"`).

    Example:
        nifti_files = prepare_spm_4d_data("/path/to/4d_pet_image.nii")
        print(nifti_files)  # ['path_to_file,1', 'path_to_file,2', ...]
    """
    nii = nib.load(file_path)
    data = nii.get_fdata()
    size = data.shape[3]  # Number of 3D volumes in the 4D image
    nifti_files = [f"'{file_path},{idx + 1}'\n" for idx in range(size)]  # Format for SPM indices
    return nifti_files


def write_matlab_file(script: str, name_with_ext: str, output_dir: str) -> str:
    """
    Writes a MATLAB script to a file.

    Args:
        script (str): The MATLAB script content as a string.
        name_with_ext (str): The filename of the MATLAB script, including the extension (e.g., `script.m`).
        output_dir (str): Directory where the MATLAB script will be saved.

    Returns:
        str: Absolute path to the created MATLAB script.

    Example:
        script_content = "disp('Hello, MATLAB!');"
        script_path = write_matlab_file(script_content, "hello.m", "/path/to/output")
        print(f"MATLAB script saved to {script_path}")
    """
    # Construct the full path for the MATLAB script file
    matlab_file_path = os.path.join(output_dir, name_with_ext)
    
    # Write the script to the specified file
    with open(matlab_file_path, "w") as fp:
        fp.write(script)
    
    return matlab_file_path
