# dvr.py
import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich import print
from petscope.kinetic_modeling.srtm2 import srtm2, region_bounds, default_bounds, calculate_weights
from petscope.utils import get_reference_region_mask, get_target_region_mask
from petscope.dynamicpet_wrapper.srtm import compute_target_region_stats

            
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


def prepare_rois(pet_data, brain_mask, roi_regions, template_path, template_name, output_dir, affine, header, verbose=True):
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
    
    if roi_regions is None:
        # Automatically select high-binding regions (top 10% within brain mask)
        mean_activity = np.mean(pet_data, axis=3)
        masked_activity = mean_activity * brain_mask
        threshold = np.percentile(masked_activity[masked_activity > 0], 90)
        high_binding_mask = (masked_activity > threshold) & brain_mask
        
        # Save high-binding ROI mask
        high_binding_mask_path = os.path.join(output_dir, "high_binding_mask.nii.gz")
        high_binding_img = nib.Nifti1Image(high_binding_mask.astype(np.int16), affine, header)
        nib.save(high_binding_img, high_binding_mask_path)
        
        # Extract mean TAC from high-binding regions
        roi_tacs = {"high_binding": np.mean(pet_data[high_binding_mask], axis=0)}
        
        if verbose:
            print(f"Automatically selected high-binding regions with {np.sum(high_binding_mask)} voxels")
    else:
        # Use specified ROIs for k2prime estimation
        roi_tacs = {}
        for roi_name in roi_regions:
            roi_mask_path = os.path.join(output_dir, f"{roi_name}_mask.nii.gz")
            # Call ROI mask creation function
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
                print(f"ROI: {roi_name}, voxel count: {np.sum(roi_mask_data)}")
    
    return {'roi_tacs': roi_tacs}


def estimate_k2prime(t_tac, ref_tac, roi_tacs, frame_durations, multstart_iter=100, verbose=True):
    """
    First pass: Estimate k2prime from ROIs using conventional SRTM.
    
    Args:
        t_tac (ndarray): Time points array
        ref_tac (ndarray): Reference region TAC
        roi_tacs (dict): Dictionary of ROI TACs
        frame_durations (list): How long each time frame lasts
        multstart_iter (int): Number of iterations for multi-start optimization
        verbose (bool): Flag to enable detailed output
        
    Returns:
        dict: Dictionary containing k2prime and ROI results
    """
    if verbose:
        print("\nFirst pass: Estimating k2prime from ROIs...")
    
    roi_results = {}
    k2prime_values = []

    for roi_name, roi_tac in roi_tacs.items():
        if verbose:
            print(f"  Fitting ROI: {roi_name}")
        
        # Get region-specific bounds or use defaults
        bounds = region_bounds.get(roi_name, default_bounds)

        weights = calculate_weights(frame_durations, roi_tac) if frame_durations is not None else np.ones_like(t_tac)
        # Use SRTM without fixed k2prime
        result = srtm2(
            t_tac=t_tac,
            reftac=ref_tac,
            roitac=roi_tac,
            k2prime=None,  # No fixed k2prime for first pass,
            weights=weights,
            multstart_iter=multstart_iter,
            printvals=verbose,
            # Use region-specific bounds
            R1_start=bounds["R1"]["start"], 
            R1_lower=bounds["R1"]["lower"], 
            R1_upper=bounds["R1"]["upper"],
            k2prime_start=bounds["k2prime"]["start"], 
            k2prime_lower=bounds["k2prime"]["lower"], 
            k2prime_upper=bounds["k2prime"]["upper"],
            bp_start=bounds["bp"]["start"], 
            bp_lower=bounds["bp"]["lower"], 
            bp_upper=bounds["bp"]["upper"]
        )
        
        roi_results[roi_name] = result
        k2prime_value = result['par']['k2prime'].values[0]
        k2prime_values.append(k2prime_value)
        
        if verbose:
            print(f"    Estimated parameters:")
            print(f"    R1 = {result['par']['R1'].values[0]:.4f}")
            print(f"    k2prime = {k2prime_value:.4f}")
            print(f"    BP = {result['par']['bp'].values[0]:.4f}")
            print(f"    DVR = {result['par']['bp'].values[0] + 1:.4f}")
    
    # Calculate global k2prime (use median to be robust to outliers)
    global_k2prime = np.median(k2prime_values)
    
    if verbose:
        print(f"\nEstimated global k2prime: {global_k2prime:.4f}")
    
    return {
        'global_k2prime': global_k2prime,
        'roi_results': roi_results,
        'k2prime_values': k2prime_values
    }


def create_parametric_images(pet_data, t_tac, ref_tac, brain_mask, global_k2prime, frame_durations, verbose=True):
    """
    Second pass: Create parametric images with fixed k2prime.
    
    Args:
        pet_data (ndarray): 4D PET data array
        t_tac (ndarray): Time points array
        ref_tac (ndarray): Reference region TAC
        brain_mask (ndarray): Brain mask array
        global_k2prime (float): Global k2prime value from first pass
        verbose (bool): Flag to enable detailed output
        
    Returns:
        dict: Dictionary containing parametric images
    """
    if verbose:
        print("\nSecond pass: Generating voxel-wise parametric images with fixed k2prime...")
    
    # Initialize output parameter volumes
    shape = pet_data.shape[:3]
    R1_img = np.zeros(shape)
    bp_img = np.zeros(shape)
    dvr_img = np.zeros(shape)
    k2a_img = np.zeros(shape)
    
    # Count masked voxels for progress tracking
    total_voxels = np.sum(brain_mask)
    voxel_counter = 0
    fail_counter = 0
    
    # Loop through voxels within the brain mask
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if brain_mask[x, y, z]:
                    voxel_counter += 1
                    
                    # Show progress periodically
                    if verbose and voxel_counter % 1000 == 0:
                        print(f"  Processed {voxel_counter}/{total_voxels} voxels ({100*voxel_counter/total_voxels:.1f}%)")
                        if fail_counter > 0:
                            print(f"  Failed fits: {fail_counter} ({100*fail_counter/voxel_counter:.1f}%)")
                    
                    # Extract voxel TAC
                    voxel_tac = pet_data[x, y, z, :]
                    
                    try:
                        weights = calculate_weights(frame_durations, voxel_tac) if frame_durations is not None else np.ones_like(t_tac)
                        # Fit SRTM2 with fixed k2prime
                        result = srtm2(
                            t_tac=t_tac,
                            reftac=ref_tac,
                            roitac=voxel_tac,
                            k2prime=global_k2prime,  # Fixed k2prime from first pass
                            multstart_iter=1,  # Use single fit for speed
                            printvals=False,
                            weights=weights,
                            R1_start=1.0, R1_lower=0.0, R1_upper=5.0,
                            bp_start=1.0, bp_lower=-0.5, bp_upper=10.0  # Allow slightly negative BP for noise
                        )
                        
                        # Store parameter values
                        R1_img[x, y, z] = result['par']['R1'].values[0]
                        bp_img[x, y, z] = result['par']['bp'].values[0]
                        k2a_img[x, y, z] = result['par']['k2a'].values[0]
                        # IMPORTANT: Explicitly calculate DVR as BP + 1
                        dvr_img[x, y, z] = result['par']['bp'].values[0] + 1.0
                        
                    except Exception as e:
                        # Handle fitting failures - use default values
                        fail_counter += 1
                        if verbose and fail_counter % 10000 == 0:
                            print(f"    Error fitting voxel ({x},{y},{z}): {e}")
                        R1_img[x, y, z] = 0
                        bp_img[x, y, z] = 0
                        k2a_img[x, y, z] = 0
                        dvr_img[x, y, z] = 1  # No binding
    
    if verbose:
        print(f"Voxel-wise fitting complete:")
        print(f"  Total voxels processed: {voxel_counter}")
        fail_percentage = 100 * fail_counter / voxel_counter if voxel_counter > 0 else 0
        print(f"  Failed fits: {fail_counter} ({fail_percentage:.1f}%)")
    
    return {
        'R1_img': R1_img,
        'bp_img': bp_img,
        'k2a_img': k2a_img,
        'dvr_img': dvr_img
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
    with open(os.path.join(output_dir, 'srtm2_k2prime.txt'), 'w') as f:
        f.write(f"Global k2prime: {global_k2prime}\n")
        for roi_name, k2prime_val in zip(roi_tacs.keys(), k2prime_values):
            f.write(f"{roi_name}: {k2prime_val}\n")
    
    # Save ROI results to CSV
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

def get_srtm2_dvr_image(pet_file, output_dir, frame_durations, template_path, template_name, reference_region, 
                        roi_regions=None, multstart_iter=100, verbose=True):
    """
    Process a complete PET study to generate DVR images using the two-pass SRTM2 approach.
    This is the main controller function that coordinates the entire process.
    
    Args:
        pet_file (str): Path to 4D PET NIFTI file
        output_dir (str): Directory for output files
        frame_durations (array_like): Duration of each frame in seconds
        template_path (str): Path to template/atlas for region definition
        template_name (str): Name of the template/atlas
        reference_region (str): Name of the reference region (e.g., 'cerebellum')
        roi_regions (list, optional): List of ROI names for k2prime estimation
        multstart_iter (int, optional): Number of multi-start iterations for ROI fitting
        verbose (bool, optional): Whether to print detailed progress information
    
    Returns:
        dict: Dictionary containing paths to output files and parameter estimates
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Prepare PET data
    pet_info = prepare_pet_data(pet_file, frame_durations, verbose)
    
    # 2. Create masks (reference region and brain)
    masks_info = create_masks(
        pet_info['pet_data'], 
        template_path, 
        template_name, 
        reference_region, 
        output_dir, 
        verbose
    )
    
    # 3. Prepare ROIs for k2prime estimation
    roi_info = prepare_rois(
        pet_info['pet_data'], 
        masks_info['brain_mask'], 
        roi_regions, 
        template_path, 
        template_name, 
        output_dir, 
        pet_info['affine'], 
        pet_info['header'], 
        verbose
    )
    
    # 4. First pass: Estimate k2prime from ROIs
    k2prime_info = estimate_k2prime(
        t_tac=pet_info['t_tac'], 
        ref_tac=masks_info['ref_tac'], 
        roi_tacs=roi_info['roi_tacs'], 
        multstart_iter=multstart_iter, 
        frame_durations=frame_durations,
        verbose=verbose,
    )
    
    # 5. Second pass: Create parametric images with fixed k2prime
    parametric_images = create_parametric_images(
        pet_data=pet_info['pet_data'], 
        t_tac=pet_info['t_tac'], 
        ref_tac= masks_info['ref_tac'], 
        brain_mask=masks_info['brain_mask'], 
        global_k2prime=k2prime_info['global_k2prime'], 
        frame_durations=frame_durations,
        verbose=verbose
    )
    
    # 6. Save results
    output_paths = save_results(
        parametric_images, 
        pet_info['affine'], 
        pet_info['header'], 
        output_dir, 
        roi_info['roi_tacs'], 
        k2prime_info['roi_results'], 
        k2prime_info['k2prime_values'], 
        k2prime_info['global_k2prime'], 
        verbose
    )
    
    # 7. Create visualization
    viz_path = create_visualization(
        parametric_images, 
        k2prime_info['global_k2prime'], 
        output_dir, 
        verbose
    )

    # 8. Compute statistics
    dvr_path = os.path.join(output_dir, 'srtm2_dvr.nii.gz')
    for target_region in roi_regions:
        compute_target_region_stats(
            dvr_path=dvr_path,
            template_path=template_path,
            template_name=template_name,
            target_region=target_region,
            output_dir=output_dir
        )

    for roi_name in roi_info['roi_tacs']:
        plot_tacs(
            t_tac=pet_info['t_tac'],
            ref_tac=masks_info['ref_tac'],
            roi_tac=roi_info['roi_tacs'][roi_name],
            output_path=os.path.join(output_dir, f'{roi_name.replace(' ', '_')}_tac_comparison.png')
        )

    if verbose:
        print(f"\n✅ SRTM2 DVR processing complete. Results saved to {output_dir}")
    
    # Return consolidated results
    return {
        'dvr_path': output_paths['dvr_path'],
        'bp_path': output_paths['bp_path'],
        'R1_path': output_paths['R1_path'],
        'k2a_path': output_paths['k2a_path'],
        'k2prime': k2prime_info['global_k2prime'],
        'roi_results': k2prime_info['roi_results'],
        'visualization_path': viz_path
    }