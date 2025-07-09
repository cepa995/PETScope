import numpy as np
import matplotlib.pyplot as plt
from rich import print

class SRTM2Validator:
    """
    Comprehensive validation framework for SRTM2 implementation
    Based on Wu & Carson (2002) and Lammertsma & Hume (1996)
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def generate_reference_tac(self, t_min, K1_ref=0.15, k2_ref=0.12, input_amplitude=2.0, input_decay=0.1):
        """
        Generate realistic reference region TAC using a simplified input function
        
        Parameters:
        -----------
        t_min : array
            Time points in minutes
        K1_ref : float
            Reference region K1 (ml/min/ml) - influx rate
        k2_ref : float  
            Reference region k2 (1/min) - efflux rate
        input_amplitude : float
            Peak amplitude of input function
        input_decay : float
            Decay rate of input function
        """
        # Create a realistic input function (exponential decay from peak)
        # This simulates the arterial input function shape
        input_func = input_amplitude * np.exp(-input_decay * t_min)
        
        # Reference region follows one-tissue compartment model
        # C_ref(t) = K1_ref * input ⊗ exp(-k2_ref * t)
        
        # Use discrete convolution
        dt = np.diff(t_min)
        if len(dt) > 0:
            step = np.mean(dt)
        else:
            step = 1.0
            
        # Exponential impulse response
        exp_response = np.exp(-k2_ref * t_min)
        
        # Convolve input with exponential response
        conv_length = min(len(input_func), len(exp_response))
        ref_tac = K1_ref * np.convolve(input_func[:conv_length], exp_response[:conv_length], mode='full')[:len(t_min)] * step
        
        # Ensure positive values and realistic shape
        ref_tac = np.maximum(ref_tac, 0.01)
        
        # Add small baseline to avoid numerical issues
        ref_tac = ref_tac + 0.1
        
        return ref_tac
    
    def generate_target_tac_srtm(self, t_min, ref_tac, R1, k2prime, BP):
        """
        Generate target TAC using the true SRTM convolution model
        This matches the mathematical approach used in your implementation
        """
        # Calculate derived parameters
        k2a = k2prime * R1 / (BP + 1)
        
        # Use convolution-based approach (similar to your srtm2_model function)
        dt = np.diff(t_min)
        if len(dt) > 0:
            step = np.mean(dt)
        else:
            step = 1.0
        
        # Create interpolated time grid for stable convolution
        interptime = np.linspace(np.min(t_min), np.max(t_min), 256)
        step_interp = interptime[1] - interptime[0]
        
        # Interpolate reference TAC
        iref = np.interp(interptime, t_min, ref_tac)
        
        # SRTM convolution components
        # Target TAC = R1 * ref_tac + conv(R1*(k2prime-k2a)*ref_tac, exp(-k2a*t))
        a = R1 * (k2prime - k2a) * iref
        b = np.exp(-k2a * interptime)
        
        # Simple convolution using np.convolve
        conv_result = np.convolve(a, b, mode='full')[:len(a)] * step_interp
        
        # Complete SRTM model
        i_outtac = R1 * iref + conv_result
        
        # Interpolate back to original time points
        outtac = np.interp(t_min, interptime, i_outtac)
        
        # Ensure no negative values
        outtac = np.maximum(outtac, 0.01 * np.max(outtac))
        
        return outtac
    
    def create_validation_scenarios(self):
        """
        Create multiple validation scenarios with different parameter combinations
        Based on literature values for various tracers
        """
        scenarios = {
            'high_binding': {'R1': 1.2, 'k2prime': 0.15, 'BP': 3.0},  # High binding region
            'medium_binding': {'R1': 1.0, 'k2prime': 0.15, 'BP': 1.5},  # Medium binding
            'low_binding': {'R1': 0.8, 'k2prime': 0.15, 'BP': 0.3},   # Low binding
            'no_binding': {'R1': 0.9, 'k2prime': 0.15, 'BP': 0.0},    # No specific binding
            'challenging': {'R1': 1.8, 'k2prime': 0.08, 'BP': 0.8},   # Challenging fit
        }
        return scenarios
    
    def add_realistic_noise(self, tac, frame_durations, noise_level=0.05, decay_constant=0.1):
        """
        Add realistic Poisson-like noise that increases with time due to decay
        
        Parameters:
        -----------
        tac : array
            Clean TAC
        frame_durations : array
            Frame durations in minutes
        noise_level : float
            Base noise level as fraction of signal
        decay_constant : float
            How quickly noise increases with time
        """
        # Ensure TAC has positive values
        tac = np.maximum(tac, 0.01 * np.max(tac))
        
        # Convert frame durations to numpy array
        frame_durations = np.array(frame_durations)
        
        # Simulate realistic count statistics
        # Noise increases as activity decreases (later frames have more noise)
        cumulative_time = np.cumsum(frame_durations)
        time_factor = 1 + decay_constant * cumulative_time / np.max(cumulative_time)
        
        # Calculate noise standard deviation
        base_noise = noise_level * tac
        noise_std = base_noise * time_factor
        
        # Ensure noise_std is always positive
        noise_std = np.maximum(noise_std, 0.001 * np.max(tac))
        
        # Add Gaussian noise (approximation of Poisson for high counts)
        noisy_tac = tac + np.random.normal(0, noise_std)
        
        # Ensure no negative values (clamp to small positive value)
        noisy_tac = np.maximum(noisy_tac, 0.01 * np.max(tac))
        
        return noisy_tac
    
    def validate_scenario(self, scenario_name, true_params, srtm2_func,
                         noise_levels=[0.02, 0.05, 0.1], n_replicates=10):
        """
        Validate one scenario with multiple noise levels
        """
        print(f"\n=== Validating Scenario: {scenario_name} ===")
        print(f"True parameters: R1={true_params['R1']:.3f}, k2prime={true_params['k2prime']:.3f}, BP={true_params['BP']:.3f}")
        
        results = {
            'noise_levels': noise_levels,
            'R1_estimates': [],
            'k2prime_estimates': [],
            'BP_estimates': [],
            'DVR_estimates': [],
            'R1_bias': [],
            'BP_bias': [],
            'DVR_bias': [],
            'R1_std': [],
            'BP_std': [],
            'DVR_std': []
        }
        
        # Generate time points (typical PET protocol)
        frame_durations = np.array([0.5, 0.5, 0.5, 0.5, 1, 1, 1, 2, 2, 2, 2, 5, 5, 5, 5, 10, 10, 10, 10])  # minutes
        t_mid = np.cumsum(frame_durations) - frame_durations/2
        
        # Generate reference TAC
        ref_tac = self.generate_reference_tac(t_mid)
        
        # Generate true target TAC
        true_target_tac = self.generate_target_tac_srtm(
            t_mid, ref_tac, 
            true_params['R1'], 
            true_params['k2prime'], 
            true_params['BP']
        )
        
        # Verify the synthetic data makes sense
        late_ratio = np.mean(true_target_tac[-3:]) / np.mean(ref_tac[-3:])
        expected_dvr = true_params['BP'] + 1
        print(f"  Synthetic data check:")
        print(f"    Expected DVR: {expected_dvr:.3f}")
        print(f"    Late-frame ratio: {late_ratio:.3f}")
        print(f"    Reference peak: {np.max(ref_tac):.2f}")
        print(f"    Target peak: {np.max(true_target_tac):.2f}")
        
        if abs(late_ratio - expected_dvr) > 0.2:
            print(f"    :warning: WARNING: Synthetic data inconsistency detected!")
            print(f"    This suggests an issue with the forward model")
        
        for noise_level in noise_levels:
            print(f"\n  Testing noise level: {noise_level:.1%}")
            
            R1_est_list = []
            k2prime_est_list = []
            BP_est_list = []
            DVR_est_list = []
            
            for rep in range(n_replicates):
                try:
                    # Add noise
                    noisy_target = self.add_realistic_noise(
                        true_target_tac, frame_durations, noise_level
                    )
                    noisy_ref = self.add_realistic_noise(
                        ref_tac, frame_durations, noise_level * 0.5  # Reference has less noise
                    )
                    
                    # Test your SRTM2 implementation
                    result = srtm2_func(
                        t_tac=t_mid,
                        reftac=noisy_ref,
                        roitac=noisy_target,
                        k2prime=None,  # Let it estimate k2prime first
                        multstart_iter=20
                    )
                    
                    R1_est = result['par']['R1'].values[0]
                    k2prime_est = result['par']['k2prime'].values[0]
                    BP_est = result['par']['bp'].values[0]
                    DVR_est = BP_est + 1
                    
                    R1_est_list.append(R1_est)
                    k2prime_est_list.append(k2prime_est)
                    BP_est_list.append(BP_est)
                    DVR_est_list.append(DVR_est)
                    
                except Exception as e:
                    print(f"    Fit failed for replicate {rep}: {e}")
                    continue
            
            if len(R1_est_list) > 0:
                # Calculate statistics
                R1_mean = np.mean(R1_est_list)
                BP_mean = np.mean(BP_est_list)
                DVR_mean = np.mean(DVR_est_list)
                
                results['R1_estimates'].append(R1_est_list)
                results['k2prime_estimates'].append(k2prime_est_list)
                results['BP_estimates'].append(BP_est_list)
                results['DVR_estimates'].append(DVR_est_list)
                
                results['R1_bias'].append((R1_mean - true_params['R1']) / true_params['R1'] * 100)
                results['BP_bias'].append((BP_mean - true_params['BP']) / max(true_params['BP'], 0.1) * 100)
                results['DVR_bias'].append((DVR_mean - (true_params['BP'] + 1)) / (true_params['BP'] + 1) * 100)
                
                results['R1_std'].append(np.std(R1_est_list))
                results['BP_std'].append(np.std(BP_est_list))
                results['DVR_std'].append(np.std(DVR_est_list))
                
                print(f"    R1: {R1_mean:.3f} ± {np.std(R1_est_list):.3f} (true: {true_params['R1']:.3f})")
                print(f"    BP: {BP_mean:.3f} ± {np.std(BP_est_list):.3f} (true: {true_params['BP']:.3f})")
                print(f"    DVR: {DVR_mean:.3f} ± {np.std(DVR_est_list):.3f} (true: {true_params['BP'] + 1:.3f})")
                
                # Check for systematic bias
                bp_bias_pct = (BP_mean - true_params['BP']) / max(true_params['BP'], 0.1) * 100
                if abs(bp_bias_pct) > 10:
                    print(f"    ⚠️ HIGH BP BIAS: {bp_bias_pct:.1f}%")
                    
            else:
                print(f"    All fits failed for noise level {noise_level}")
        
        self.validation_results[scenario_name] = results
        return results
    
    def run_full_validation(self, srtm2_func):
        """
        Run complete validation suite
        """
        scenarios = self.create_validation_scenarios()
        
        print("="*60)
        print("SRTM2 VALIDATION SUITE")
        print("="*60)
        
        for scenario_name, true_params in scenarios.items():
            self.validate_scenario(scenario_name, true_params, srtm2_func)
        
        # Generate summary report
        self.generate_validation_report()
    
    def generate_validation_report(self):
        """
        Generate comprehensive validation report with plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        scenarios = list(self.validation_results.keys())
        noise_levels = self.validation_results[scenarios[0]]['noise_levels']
        
        # Plot bias for each parameter
        for i, param in enumerate(['R1_bias', 'BP_bias', 'DVR_bias']):
            ax = axes[0, i]
            for scenario in scenarios:
                bias = self.validation_results[scenario][param]
                ax.plot(noise_levels, bias, 'o-', label=scenario)
            ax.set_xlabel('Noise Level')
            ax.set_ylabel(f'{param.split("_")[0]} Bias (%)')
            ax.set_title(f'{param.split("_")[0]} Bias vs Noise')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # Plot standard deviation for each parameter  
        for i, param in enumerate(['R1_std', 'BP_std', 'DVR_std']):
            ax = axes[1, i]
            for scenario in scenarios:
                std = self.validation_results[scenario][param]
                ax.plot(noise_levels, std, 'o-', label=scenario)
            ax.set_xlabel('Noise Level')
            ax.set_ylabel(f'{param.split("_")[0]} Standard Deviation')
            ax.set_title(f'{param.split("_")[0]} Precision vs Noise')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('srtm2_validation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        for scenario in scenarios:
            results = self.validation_results[scenario]
            print(f"\nScenario: {scenario}")
            print(f"  R1 bias range: {min(results['R1_bias']):.1f}% to {max(results['R1_bias']):.1f}%")
            print(f"  BP bias range: {min(results['BP_bias']):.1f}% to {max(results['BP_bias']):.1f}%")
            print(f"  DVR bias range: {min(results['DVR_bias']):.1f}% to {max(results['DVR_bias']):.1f}%")

def test_srtm2():
    """
    Example of how to use the validator with your SRTM2 function
    """
    from petscope.kinetic_modeling.srtm2 import srtm2  
    
    validator = SRTM2Validator()
    
    # Wrapper to match expected interface
    def srtm2_wrapper(t_tac, reftac, roitac, k2prime, multstart_iter):
        return srtm2(
            t_tac=t_tac,
            reftac=reftac, 
            roitac=roitac,
            k2prime=k2prime,
            multstart_iter=multstart_iter,
            printvals=False
        )
    
    # Run validation
    validator.run_full_validation(srtm2_wrapper)

if __name__ == "__main__":
    test_srtm2()