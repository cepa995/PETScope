import os
from typing import List

def get_realignment_script(pet3d_volumes: List[str], output_dir: str, script_name: str) -> str:
    """Create MATLAB script for data realignment"""
    script_path = os.path.join(output_dir, script_name)
    with open(script_path, "w") as fp:
        fp.writelines(
            [
                # "spm('Defaults', 'fMRI');\n",
                "spm_jobman('initcfg');\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.data = {",
                "{\n",
            ]
            + pet3d_volumes
            + [
                "}\n",
                "};\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.sep = 4;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.rtm = 1;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.interp = 2;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.weight = '';\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.roptions.which = [2 1];\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.roptions.interp = 4;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.roptions.mask = 1;\n",
                "matlabbatch{1}.spm.spatial.realign.estwrite.roptions.prefix = 'r';\n",
                "spm_jobman('run',matlabbatch);\n",
            ]
        )
    return script_path
