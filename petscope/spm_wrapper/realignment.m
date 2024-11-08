spm_jobman('initcfg');
matlabbatch{1}.spm.spatial.realign.estwrite.data = {
{
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,1'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,2'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,3'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,4'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,5'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,6'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,7'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,8'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,9'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,10'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,11'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,12'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,13'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,14'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,15'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,16'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,17'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,18'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,19'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,20'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,21'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,22'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,23'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,24'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,25'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,26'
'/neuro/stefan/workspace/PETScope-CLI-Test/pet_4d_rsa.nii,27'
}
};
matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;
matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.sep = 4;
matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;
matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.rtm = 1;
matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.interp = 2;
matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];
matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.weight = '';
matlabbatch{1}.spm.spatial.realign.estwrite.roptions.which = [2 1];
matlabbatch{1}.spm.spatial.realign.estwrite.roptions.interp = 4;
matlabbatch{1}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];
matlabbatch{1}.spm.spatial.realign.estwrite.roptions.mask = 1;
matlabbatch{1}.spm.spatial.realign.estwrite.roptions.prefix = 'r';
spm_jobman('run',matlabbatch);
