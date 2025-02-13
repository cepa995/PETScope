import pytest
import numpy as np
import pandas as pd

@pytest.mark.dependency()
def test_srtm2(srtm2_test_args):
    """ Tests Python implementation of SRTM2 motived by 
    https://github.com/mathesong/kinfitr """
    from petscope.kinetic_modeling.srtm2 import srtm2

    # Load test data from CSV
    simRef_df = pd.read_csv(srtm2_test_args["file_path"])
    
    # Parse SRTM2 Input arguments
    t_tac = simRef_df['Times'].values
    reftac = simRef_df['Reference'].values
    roitac = simRef_df['ROI1'].values
    weights = simRef_df['Weights'].values
    
    # Run function
    output = srtm2(
        t_tac=t_tac,
        reftac=reftac,
        roitac=roitac,
        weights=weights,
        multstart_iter=1
    )
    print(output)

    # Assertions
    assert isinstance(output, dict)
    assert 'par' in output and isinstance(output['par'], pd.DataFrame)
    assert 'par_se' in output and isinstance(output['par_se'], pd.DataFrame)
    assert 'fit' in output and isinstance(output['fit'], dict)
    assert 'weights' in output and isinstance(output['weights'], np.ndarray)
    assert 'tacs' in output and isinstance(output['tacs'], pd.DataFrame)
    assert 'model' in output and output['model'] == 'srtm2'
    
    # Check key parameter values
    expected_params = np.array([1.003753, 0.993378, 0.000509, 0.9966])
    np.testing.assert_almost_equal(output['par'].values.flatten(), expected_params, decimal=5)
    
    # Check TACs DataFrame structure
    assert list(output['tacs'].columns) == ['Time', 'Reference', 'Target', 'Target_fitted']
    assert len(output['tacs']) == len(t_tac)
