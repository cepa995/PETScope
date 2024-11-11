import os
import nibabel as nib
from typing import List

def get_realignment_script(pet3d_volumes: List[str], output_dir: str, script_name: str) -> str:
    """
    Creates a MATLAB script for PET image data realignment using SPM.

    :param pet3d_volumes: List of 3D PET volume file paths to be included in the realignment.
    :param output_dir: Directory where the MATLAB script will be created.
    :param script_name: Name of the MATLAB script file to be created.
    :returns: The path to the generated MATLAB script.
    """
    # Construct the full path for the script
    script_path = os.path.join(output_dir, script_name)
    
    # Open the file for writing the MATLAB script
    with open(script_path, "w") as fp:
        # Write the MATLAB commands to configure the realignment batch
        fp.writelines(
            [
                # Initialize the SPM job manager
                "spm_jobman('initcfg');\n",
                
                # Define the input data for realignment
                "matlabbatch{1}.spm.spatial.realign.estwrite.data = {\n",
                "{\n",
            ]
            + pet3d_volumes  # Add the list of PET 3D volume paths
            + [
                "}\n",
                "};\n",
                
                # Set options for estimation and writing of the realignment
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.sep = 4;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.rtm = 1;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.interp = 2;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.weight = '';\n",
                
                # Set options for reslicing the images
                "matlabbatch{1}.spm.spatial.realign.estwrite.roptions.which = [2 1];\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.roptions.interp = 4;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.roptions.mask = 1;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.roptions.prefix = 'r';\n",
                
                # Run the batch job
                "spm_jobman('run',matlabbatch);\n",
            ]
        )
        
    # Return the path to the created script
    return script_path

def prepare_spm_4d_data(file_path):
    """
    Replicates output from spm_select function

    :param file_path: absolute path to 4D PET image
    :return a list of PET files and their indicies such as in spm_select
    """
    nii = nib.load(file_path)
    data = nii.get_fdata()
    size = data.shape[3]
    nifti_files = [None] * size
    for idx in range(size):
        # SPM indexes start from 1
        if (idx + 1) < 10:
            nifti_files[idx] = "'" + file_path + "," + str(idx + 1) + "'\n"
        else:
            nifti_files[idx] = "'" + file_path + "," + str(idx + 1) + "'\n"
    return nifti_files


def write_matlab_file(script, name_with_ext, output_dir):
    """
    Write a matlab file as the output.

    :param script: matlab script as a string
    :param name_with_ext: matlab script filename with extension
    :return matlab+_file: absolute path to matlab file
    """
    matlab_file_path = os.path.join(output_dir, name_with_ext)
    with open(matlab_file_path, "w") as fp:
        fp.write(script)

    return matlab_file_path