import os
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from petscope.utils import get_reference_region_mask, get_target_region_mask


def calculate_weights(frame_durations, tac_values):
    """Calculate weights based on frame duration and decay-corrected counts."""
    return np.sqrt(frame_durations) / np.sqrt(np.maximum(tac_values, 0.01))

def kinfit_convolve(a, b, step):
    """
    Convolve two signals a and b using a fixed time step.
    This approximates the continuous convolution using a Riemann sum.
    """
    # The full convolution is computed; we then take only the first len(a) points
    conv = np.convolve(a, b, mode='full')[:len(a)] * step
    return conv

def prepare_pet_data(pet_file, frame_durations, verbose=True):
    """
    Load PET data and prepare time information.
    
    Args:
        pet_file (str): Path to the 4D PET NIFTI file
        frame_durations (array_like): Duration of each frame in seconds
        verbose (bool): Flag to enable detailed output
        
    Returns:
        dict: Dictionary containing PET data and related information
    """
    if verbose:
        print(f"Loading and preparing PET data from: {pet_file}")
    
    # Load 4D PET data
    pet_img = nib.load(pet_file)
    pet_data = pet_img.get_fdata()
    
    # Calculate mid-frame times (convert to minutes)
    frame_durations_min = np.array(frame_durations) / 60.0
    t_tac = np.cumsum(frame_durations_min) - frame_durations_min/2
    
    if verbose:
        print(f"PET data shape: {pet_data.shape}")
        print(f"Number of frames: {len(t_tac)}")
        print(f"Total scan duration: {np.sum(frame_durations_min):.2f} minutes")
    
    return {
        'pet_img': pet_img,
        'pet_data': pet_data,
        'header': pet_img.header,
        'affine': pet_img.affine,
        't_tac': t_tac
    }

def fix_multstartpars(start, lower, upper, multstart_iter, multstart_lower=None, multstart_upper=None):
    """
    Check and fix multstart parameters.
    
    Checks missing multstart boundaries, verifies if some parameters should not be iterated,
    and ensures correct lengths for multstart bounds.
    
    Parameters:
    start : dict
        Original starting values without multstart.
    lower : dict
        Lower fitting bounds.
    upper : dict
        Upper fitting bounds.
    multstart_iter : int or list
        Number of iterations.
    multstart_lower : dict, optional
        Lower starting bounds.
    multstart_upper : dict, optional
        Upper starting bounds.
    
    Returns:
    dict
        A dictionary containing adjusted multstart_lower and multstart_upper bounds.
    """
    if isinstance(multstart_iter, int) or len(multstart_iter) == len(start):
        multstart_l = lower.copy()
        multstart_u = upper.copy()
        
        parnames = start.keys()
        
        # Adding multstart boundaries
        if multstart_lower is not None:
            if not isinstance(multstart_lower, dict) or not all(k in parnames for k in multstart_lower.keys()):
                raise ValueError("multstart_lower should be a dictionary with keys matching the parameters to be fitted")
            multstart_lower = {**multstart_l, **multstart_lower}
        else:
            multstart_lower = multstart_l
        
        if multstart_upper is not None:
            if not isinstance(multstart_upper, dict) or not all(k in parnames for k in multstart_upper.keys()):
                raise ValueError("multstart_upper should be a dictionary with keys matching the parameters to be fitted")
            multstart_upper = {**multstart_u, **multstart_upper}
        else:
            multstart_upper = multstart_u
        
        # No multstart for some variables
        if isinstance(multstart_iter, list) and any(i == 1 for i in multstart_iter):
            non_iterable = [i for i, val in enumerate(multstart_iter) if val == 1]
            for idx in non_iterable:
                key = list(start.keys())[idx]
                multstart_lower[key] = start[key]
                multstart_upper[key] = start[key]
        
        # Ensure correct order
        multstart_lower = {k: multstart_lower[k] for k in start.keys()}
        multstart_upper = {k: multstart_upper[k] for k in start.keys()}
        
        return {
            "multstart_lower": multstart_lower,
            "multstart_upper": multstart_upper
        }
    else:
        raise ValueError("multstart_iter should be of length 1 or the same length as the number of parameters")


def create_masks(pet_data, template_path, template_name, reference_region, output_dir, verbose=True):
    """
    Create reference region and brain masks.
    
    Args:
        pet_data (ndarray): 4D PET data array
        template_path (str): Path to the template/atlas
        template_name (str): Name of the template/atlas
        reference_region (str): Name of the reference region
        output_dir (str): Directory to save outputs
        verbose (bool): Flag to enable detailed output
        
    Returns:
        dict: Dictionary containing masks and reference TAC
    """
    if verbose:
        print("Creating reference and brain masks...")
    
    # Create reference region mask
    reference_mask_path = os.path.join(output_dir, f"{reference_region}_mask.nii.gz")
    if verbose:
        print(f"Creating reference region mask for {reference_region}")
    
    _ = get_reference_region_mask(
        template_path=template_path,
        template_name=template_name,
        reference_name=reference_region,
        mask_out=reference_mask_path
    )
    
    # Extract reference TAC
    if os.path.exists(reference_mask_path):
        # Load reference region mask
        ref_mask_img = nib.load(reference_mask_path)
        ref_mask = ref_mask_img.get_fdata() > 0
        
        # Extract mean TAC from reference region
        ref_tac = np.mean(pet_data[ref_mask], axis=0)
        
        if verbose:
            print(f"Reference region: {reference_region}")
            print(f"Reference region voxel count: {np.sum(ref_mask)}")
    else:
        raise FileNotFoundError(f"Reference region mask could not be created: {reference_mask_path}")
    
    # Load or create a brain mask for limiting processing
    brain_mask_path = os.path.join(output_dir, "brain_mask.nii.gz")
    if os.path.exists(template_path):
        # Use template as brain mask if available
        template_img = nib.load(template_path)
        brain_mask = template_img.get_fdata() > 0
        # Save brain mask
        affine = template_img.affine
        header = template_img.header
        brain_mask_img = nib.Nifti1Image(brain_mask.astype(np.int16), affine, header)
        nib.save(brain_mask_img, brain_mask_path)
    else:
        # Create simple brain mask from PET activity
        brain_mask = np.mean(pet_data, axis=3) > np.mean(ref_tac) * 0.2
        # Save brain mask
        affine = ref_mask_img.affine
        header = ref_mask_img.header
        brain_mask_img = nib.Nifti1Image(brain_mask.astype(np.int16), affine, header)
        nib.save(brain_mask_img, brain_mask_path)
    
    if verbose:
        print(f"Brain mask voxel count: {np.sum(brain_mask)}")
    
    return {
        'ref_mask': ref_mask,
        'ref_tac': ref_tac,
        'brain_mask': brain_mask
    }


def prepare_rois(pet_data, brain_mask, roi_regions, template_path, template_name, 
                      output_dir, affine, header, t_tac, verbose=True):
    """
    Prepare ROIs for k2prime estimation.
    
    Args:
        pet_data (ndarray): 4D PET data array
        brain_mask (ndarray): Brain mask array
        roi_regions (list or None): List of ROI names or None for automatic selection
        template_path (str): Path to the template/atlas
        template_name (str): Name of the template/atlas
        output_dir (str): Directory to save outputs
        affine (ndarray): Affine transformation matrix
        header (object): NIFTI header
        verbose (bool): Flag to enable detailed output
        
    Returns:
        dict: Dictionary containing ROI TACs
    """
    if verbose:
        print("Preparing ROIs for k2prime estimation...")

    # Use specified ROIs for k2prime estimation
    roi_tacs = {}
    for roi_name in roi_regions:
        roi_mask_path = os.path.join(output_dir, f"{roi_name}_mask.nii.gz")
        _ = get_target_region_mask(
            template_path=template_path,
            template_name=template_name,
            target_name=roi_name,
            mask_out=roi_mask_path
        )
        roi_mask_img = nib.load(roi_mask_path)
        roi_mask_data = roi_mask_img.get_fdata() > 0
        
        # Extract mean TAC from ROI
        roi_tacs[roi_name] = np.mean(pet_data[roi_mask_data], axis=0)
        
        if verbose:
            print(f"  ROI: {roi_name}, voxel count: {np.sum(roi_mask_data)}")
    
    return {'roi_tacs': roi_tacs}


def prepare_roi_masks(brain_mask, roi_regions, template_path, template_name, 
                      output_dir, verbose=True) -> Dict:
    """
    Prepare ROI masks for voxel-wise k2prime estimation.
    
    Args:
        brain_mask (ndarray): Brain mask array
        roi_regions (list): List of ROI names
        template_path (str): Path to template/atlas
        template_name (str): Name of template/atlas
        output_dir (str): Directory for output files
        affine (ndarray): Affine transformation matrix
        header (nibabel header): NIfTI header
        verbose (bool): Flag to enable detailed output
        
    Returns:
        dict: Dictionary containing ROI masks
    """
    if verbose:
        print("\nPreparing ROI masks for voxel-wise k2prime estimation...")
    
    roi_masks = {}
    for roi_name in roi_regions:
        if verbose:
            print(f"  Loading mask for {roi_name}...")
        
        # Create a target region mask from a specified template
        target_mask_path = os.path.join(output_dir, f"{roi_name}.nii.gz")
        roi_mask_nii = get_target_region_mask(
            template_path=template_path,
            template_name=template_name,
            target_name=roi_name,
            mask_out=target_mask_path
        )

        # Ensure mask is within brain
        roi_mask = roi_mask_nii.get_fdata()

        roi_masks[roi_name] = roi_mask
        if verbose:
            print(f"    {roi_name}: {np.sum(roi_mask)} voxels")
    
    return {
        "roi_masks": roi_masks
    }

def save_results(parametric_images, affine, header, output_dir, roi_tacs, roi_results, 
                 k2prime_values, global_k2prime, verbose=True):
    """
    Save parametric images, ROI results, and visualizations.
    
    Args:
        parametric_images (dict): Dictionary of parametric images
        affine (ndarray): Affine transformation matrix
        header (object): NIFTI header
        output_dir (str): Directory to save outputs
        roi_tacs (dict): Dictionary of ROI TACs
        roi_results (dict): Dictionary of ROI fitting results
        k2prime_values (list): List of k2prime values from ROIs
        global_k2prime (float): Global k2prime value
        verbose (bool): Flag to enable detailed output
        
    Returns:
        dict: Dictionary of output file paths
    """
    if verbose:
        print("\nSaving results...")
    
    # Create output paths
    dvr_path = os.path.join(output_dir, 'srtm2_dvr.nii.gz')
    bp_path = os.path.join(output_dir, 'srtm2_bp.nii.gz')
    R1_path = os.path.join(output_dir, 'srtm2_R1.nii.gz')
    k2a_path = os.path.join(output_dir, 'srtm2_k2a.nii.gz')
    
    # Save parametric images
    dvr_nii = nib.Nifti1Image(parametric_images['dvr_img'], affine, header)
    nib.save(dvr_nii, dvr_path)
    
    bp_nii = nib.Nifti1Image(parametric_images['bp_img'], affine, header)
    nib.save(bp_nii, bp_path)
    
    R1_nii = nib.Nifti1Image(parametric_images['R1_img'], affine, header)
    nib.save(R1_nii, R1_path)
    
    k2a_nii = nib.Nifti1Image(parametric_images['k2a_img'], affine, header)
    nib.save(k2a_nii, k2a_path)
    
    # Save k2prime value as a text file
    if roi_tacs != None:
        with open(os.path.join(output_dir, 'srtm2_k2prime.txt'), 'w') as f:
            f.write(f"Global k2prime: {global_k2prime}\n")
            for roi_name, k2prime_val in zip(roi_tacs.keys(), k2prime_values):
                f.write(f"{roi_name}: {k2prime_val}\n")
    
        # Save ROI results to CSV
        roi_tacs = {k:v for k,v in roi_tacs.items() if k in roi_results} # In case some k2 prime is below given threshold we have to EXCLUDE the TAC for that ROI
        roi_results_df = pd.DataFrame({
            'ROI': list(roi_tacs.keys()),
            'R1': [roi_results[roi]['par']['R1'].values[0] for roi in roi_tacs.keys()],
            'k2prime': [roi_results[roi]['par']['k2prime'].values[0] for roi in roi_tacs.keys()],
            'BP': [roi_results[roi]['par']['bp'].values[0] for roi in roi_tacs.keys()],
            'DVR': [roi_results[roi]['par']['bp'].values[0] + 1.0 for roi in roi_tacs.keys()],
        })
        roi_results_df.to_csv(os.path.join(output_dir, 'srtm2_roi_results.csv'), index=False)
        
    if verbose:
        print(f"Files saved:")
        print(f"  - DVR image: {dvr_path}")
        print(f"  - BP image: {bp_path}")
        print(f"  - R1 image: {R1_path}")
        print(f"  - k2a image: {k2a_path}")
        print(f"  - ROI results: {os.path.join(output_dir, 'srtm2_roi_results.csv')}")
    
    return {
        'dvr_path': dvr_path,
        'bp_path': bp_path,
        'R1_path': R1_path,
        'k2a_path': k2a_path
    }

def create_visualization(parametric_images, global_k2prime, output_dir, verbose=True):
    """
    Create visualization of parametric images.
    
    Args:
        parametric_images (dict): Dictionary of parametric images
        global_k2prime (float): Global k2prime value
        output_dir (str): Directory to save outputs
        verbose (bool): Flag to enable detailed output
        
    Returns:
        str: Path to visualization file
    """
    if verbose:
        print("Creating visualization...")
    
    # Create a quick visualization of the DVR map (mid-axial slice)
    shape = parametric_images['dvr_img'].shape
    z_mid = shape[2] // 2
    
    plt.figure(figsize=(10, 8))
    plt.imshow(parametric_images['dvr_img'][:, :, z_mid].T, cmap='hot', vmin=0.8, vmax=3.0)
    plt.colorbar(label='DVR')
    plt.title(f'SRTM2 DVR Map (Axial Slice, k2prime={global_k2prime:.4f})')
    
    viz_path = os.path.join(output_dir, 'srtm2_dvr_visualization.png')
    plt.savefig(viz_path, dpi=150)
    plt.close()
    
    if verbose:
        print(f"Visualization saved to: {viz_path}")
    
    return viz_path

def plot_tacs(t_tac, ref_tac, roi_tac, roi_name="Hippocampus", ref_name="Cerebellum", output_path=None):
    """
    Plot time-activity curves for visual inspection.
    
    Args:
        t_tac (array): Time points
        ref_tac (array): Reference region TAC
        roi_tac (array): ROI TAC
        roi_name (str): Name of the ROI
        ref_name (str): Name of the reference region
        output_path (str, optional): If provided, save the plot to this path
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t_tac, ref_tac, 'bo-', label=f'{ref_name} (Reference)')
    plt.plot(t_tac, roi_tac, 'ro-', label=f'{roi_name} (Target)')
    
    # Calculate ratio of target to reference
    ratio = roi_tac / ref_tac
    plt.plot(t_tac, ratio, 'g--', label='Target/Reference Ratio')
    
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Time (min)')
    plt.ylabel('Activity')
    plt.title(f'Time-Activity Curves: {roi_name} vs {ref_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"TAC plot saved to: {output_path}")
    else:
        plt.show()
    plt.close()
    
def diagnostic_plots(parametric_images, global_k2prime, roi_results, output_dir, verbose=True):
    """
    IMPROVEMENT: Create comprehensive diagnostic plots for SRTM2 validation.
    """
    if verbose:
        print("Creating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Get middle slice for visualization
    shape = parametric_images['dvr_img'].shape
    z_mid = shape[2] // 2
    
    # 1. DVR distribution
    valid_dvr = parametric_images['dvr_img'][parametric_images['dvr_img'] > 0]
    axes[0, 0].hist(valid_dvr, bins=50, alpha=0.7, density=True)
    axes[0, 0].axvline(np.median(valid_dvr), color='red', linestyle='--', label=f'Median: {np.median(valid_dvr):.2f}')
    axes[0, 0].set_xlabel('DVR')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('DVR Distribution')
    axes[0, 0].legend()
    
    # 2. BP distribution
    valid_bp = parametric_images['bp_img'][parametric_images['dvr_img'] > 0]
    axes[0, 1].hist(valid_bp, bins=50, alpha=0.7, density=True)
    axes[0, 1].axvline(np.median(valid_bp), color='red', linestyle='--', label=f'Median: {np.median(valid_bp):.2f}')
    axes[0, 1].set_xlabel('BP')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('BP Distribution')
    axes[0, 1].legend()
    
    # 3. R1 vs BP correlation
    axes[0, 2].scatter(parametric_images['R1_img'][parametric_images['dvr_img'] > 0], 
                      valid_bp, alpha=0.1, s=1)
    axes[0, 2].set_xlabel('R1')
    axes[0, 2].set_ylabel('BP')
    axes[0, 2].set_title('R1 vs BP Correlation')
    
    # 4. DVR image
    im1 = axes[1, 0].imshow(parametric_images['dvr_img'][:, :, z_mid].T, 
                           cmap='hot', vmin=0.8, vmax=3.0)
    axes[1, 0].set_title(f'DVR Map (k2prime={global_k2prime:.4f})')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # 5. BP image
    im2 = axes[1, 1].imshow(parametric_images['bp_img'][:, :, z_mid].T, 
                           cmap='viridis', vmin=-0.5, vmax=5.0)
    axes[1, 1].set_title('BP Map')
    plt.colorbar(im2, ax=axes[1, 1])
    
    # 6. k2prime values from ROIs
    print(roi_results)
    if roi_results:
        roi_names = list(roi_results.keys())
        k2prime_vals = [roi_results[roi]['par']['k2prime'].values[0] for roi in roi_names]
        axes[1, 2].bar(range(len(roi_names)), k2prime_vals)
        axes[1, 2].axhline(global_k2prime, color='red', linestyle='--', 
                          label=f'Global k2prime: {global_k2prime:.4f}')
        axes[1, 2].set_xticks(range(len(roi_names)))
        axes[1, 2].set_xticklabels(roi_names, rotation=45)
        axes[1, 2].set_ylabel('k2prime')
        axes[1, 2].set_title('ROI k2prime Values')
        axes[1, 2].legend()
    
    plt.tight_layout()
    diag_path = os.path.join(output_dir, 'srtm2_comprehensive_diagnostics.png')
    plt.savefig(diag_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Comprehensive diagnostics saved to: {diag_path}")
    
    return diag_path