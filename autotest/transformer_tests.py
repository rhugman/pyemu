import os
import sys
import shutil
import pytest
import numpy as np
import pandas as pd
import platform
sys.path.append("..")
import pyemu
import pyemu.emulators

def test_base_transformer():
    """Test the BaseTransformer abstract class functionality"""
    bt = pyemu.emulators.BaseTransformer()
    
    # fit should return self
    assert bt.fit(None) is bt
    
    # fit_transform should call fit and transform
    with pytest.raises(NotImplementedError):
        bt.fit_transform(None)
    
    # transform should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        bt.transform(None)
    
    # inverse_transform should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        bt.inverse_transform(None)

def test_log10_transformer():
    """Test the Log10Transformer functionality"""
    # Create test dataframe with positive and negative values
    df = pd.DataFrame({
        'pos': [1, 10, 100, 1000],
        'zero': [0, 0.1, 0.01, 0.001],
        'neg': [-1, -10, -100, -1000]
    })
    
    # Initialize and test transformer
    lt = pyemu.emulators.Log10Transformer()
    
    # Transform data
    transformed = lt.transform(df)
    
    # Check that positive values are properly transformed
    np.testing.assert_allclose(
        transformed['pos'].values,
        np.log10(df['pos'].values)
    )
    
    # Check that zeros/small values are handled correctly
    assert not np.any(np.isinf(transformed['zero'].values))
    
    # Check that negative values are handled correctly
    assert not np.any(np.isnan(transformed['neg'].values))
    
    # Test inverse transform
    back_transformed = lt.inverse_transform(transformed)
    
    # Check that we get back very close to original values
    np.testing.assert_allclose(
        back_transformed['pos'].values, 
        df['pos'].values
    )
    
    # For zero/very small values
    np.testing.assert_allclose(
        back_transformed['zero'].values,
        df['zero'].values ,
        rtol=1e-6
    )
    
    # For negative values
    np.testing.assert_allclose(
        back_transformed['neg'].values,
        df['neg'].values ,
        rtol=1e-6
    )

def test_log10_transform_uses_fitted_shift_on_new_frame():
    """Regression: fit() learns a per-column shift; transform() on a second,
    positive-min frame must use the FITTED shift (not re-learn it). The same
    value transforms identically whether it appears in the training frame or a
    new frame, and self.shifts is unchanged after transforming new data."""
    # Training column has min <= 0 -> a nonzero shift is learned on fit.
    train = pd.DataFrame({'a': [-5.0, 0.0, 10.0]})
    t = pyemu.emulators.Log10Transformer(columns=['a'])
    t.fit(train)

    fitted_shift = t.shifts['a']
    assert fitted_shift > 0  # -(-5) + 1e-6

    # The value 10.0 appears in the training frame; transform it there.
    train_transformed = t.transform(train)
    train_val_10 = train_transformed['a'].iloc[2]

    # A second frame with a *positive* min (so a re-learned shift would be 0).
    new_frame = pd.DataFrame({'a': [10.0, 20.0, 30.0]})
    new_transformed = t.transform(new_frame)

    # Same input value -> same output, because the FITTED shift is used.
    assert np.isclose(new_transformed['a'].iloc[0], train_val_10)
    np.testing.assert_allclose(
        new_transformed['a'].values,
        np.log10(new_frame['a'].values + fitted_shift),
    )

    # Transforming new data must NOT have overwritten the fitted shift.
    assert t.shifts['a'] == fitted_shift


def test_log10_roundtrip_survives_intervening_transform():
    """Regression (state corruption): inverse_transform(transform(train)) must
    round-trip, AND must still round-trip after an intervening transform() on a
    different positive-min frame. Pre-fix the intervening call overwrote the
    fitted shift and broke the round-trip."""
    train = pd.DataFrame({'a': [-5.0, 0.0, 10.0]})
    t = pyemu.emulators.Log10Transformer(columns=['a'])
    t.fit(train)

    # Transform train ONCE and keep the result.
    transformed = t.transform(train)

    # Round-trip before any intervening call.
    inversed = t.inverse_transform(transformed)
    np.testing.assert_allclose(inversed['a'].values, train['a'].values, atol=1e-5)

    # Intervening transform on a different, positive-min frame. Pre-fix this
    # re-learns a 0 shift and overwrites the stored training shift, so the
    # SAME already-transformed train data no longer inverts correctly.
    other = pd.DataFrame({'a': [10.0, 20.0, 30.0]})
    _ = t.transform(other)

    # Invert the SAME transformed train data; the stored shift must be intact.
    inversed2 = t.inverse_transform(transformed)
    np.testing.assert_allclose(inversed2['a'].values, train['a'].values, atol=1e-5)


def test_log10_below_fitted_domain_raises():
    """Regression: transforming a value below the fitted domain
    (X[col] + shift <= 0) must raise ValueError."""
    train = pd.DataFrame({'a': [-5.0, 0.0, 10.0]})
    t = pyemu.emulators.Log10Transformer(columns=['a'])
    t.fit(train)

    # fitted shift is ~5; a value of -6 gives -6 + 5 + 1e-6 < 0 -> out of domain.
    below_domain = pd.DataFrame({'a': [-6.0]})
    with pytest.raises(ValueError):
        t.transform(below_domain)


def test_log10_bare_transform_autofits():
    """Backward compat: calling transform() without an explicit fit() still
    works (auto-fit on first use)."""
    df = pd.DataFrame({'a': [1.0, 10.0, 100.0]})
    t = pyemu.emulators.Log10Transformer(columns=['a'])

    # No fit() call; transform must auto-fit and produce log10 values.
    result = t.transform(df)
    np.testing.assert_allclose(result['a'].values, [0.0, 1.0, 2.0], atol=1e-10)
    assert t.shifts['a'] == 0


def test_row_wise_minmax_scaler():
    """Test the RowWiseMinMaxScaler functionality"""
    # Test data
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [10, 20, 30, 40],
        'c': [100, 200, 300, 400]
    })
    
    # Initialize scaler
    scaler = pyemu.emulators.RowWiseMinMaxScaler()
    
    # Fit and transform
    transformed = scaler.fit_transform(df)
    
    # Check each row is scaled to [0, 1]
    for i in range(len(df)):
        row_min = transformed.iloc[i].min()
        row_max = transformed.iloc[i].max()
        assert np.isclose(row_min, -1.0)
        assert np.isclose(row_max, 1.0)
    
    # Test inverse transform
    back_transformed = scaler.inverse_transform(transformed)
    
    # Check we get back original values
    np.testing.assert_allclose(back_transformed.values, df.values)

def test_normal_score_transformer():
    """Test the NormalScoreTransformer functionality"""
    # Create test data with various distributions
    rng = np.random.RandomState(42)
    n = 200
    
    # Uniform data
    uniform_data = rng.uniform(0, 10, n)
    
    # Log-normal data
    lognormal_data = np.exp(rng.normal(0, 1, n))
    
    # Bimodal data
    bimodal_data = np.concatenate([
        rng.normal(-3, 1, n//2),
        rng.normal(3, 1, n//2)
    ])
    
    df = pd.DataFrame({
        'uniform': uniform_data,
        'lognormal': lognormal_data,
        'bimodal': bimodal_data
    })
    
    # Initialize transformer
    nst = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=False)
    
    # Transform data
    transformed = nst.fit_transform(df)
    
    # Check transformed distributions are more normal
    # For each column, check skewness and kurtosis are closer to normal
    for col in df.columns:
        # Calculate statistics of original and transformed data
        orig_skew = skewness(df[col].values)
        trans_skew = skewness(transformed[col].values)
        
        orig_kurt = kurtosis(df[col].values)
        trans_kurt = kurtosis(transformed[col].values)
        
        # Transformed data should have skewness closer to 0
        assert abs(trans_skew) < abs(orig_skew) or np.isclose(abs(trans_skew), 0, atol=0.5)
        
        # Transformed data should have kurtosis closer to 3 (normal distribution)
        assert abs(trans_kurt - 3) < abs(orig_kurt - 3) or np.isclose(trans_kurt, 3, atol=1.0)
    
    # Test inverse transform
    back_transformed = nst.inverse_transform(transformed)
    
    # Check we get back close to original values 
    # (not exact due to binning and smoothing)
    np.testing.assert_allclose(
        back_transformed.values, 
        df.values,
        rtol=0.1,
        atol=0.1
    )
    
    # Test with quadratic extrapolation
    nst_quad = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
    transformed_quad = nst_quad.fit_transform(df)
    
    # Create data outside the original range for extrapolation test
    # Transform should not fail for out-of-range values when using quadratic extrapolation
    extreme_transformed = transformed_quad.copy()
    extreme_transformed.loc[0] = transformed_quad.min() - 1
    extreme_transformed.loc[1] = transformed_quad.max() + 1
    
    back_extreme = nst_quad.inverse_transform(extreme_transformed)
    assert not np.any(np.isnan(back_extreme.values))
    assert not np.any(np.isinf(back_extreme.values))

def test_transformer_pipeline():
    """Test the TransformerPipeline functionality"""
    # Create test data
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [10, 20, 30, 40],
        'c': [100, 200, 300, 400]
    })
    
    # Create pipeline with multiple transformers
    pipeline = pyemu.emulators.TransformerPipeline()
    
    # Add log transformer for all columns
    log_trans = pyemu.emulators.Log10Transformer()
    pipeline.add(log_trans)
    
    # Add row-wise min-max scaler for specific columns
    minmax_trans = pyemu.emulators.RowWiseMinMaxScaler()
    pipeline.add(minmax_trans, columns=['a', 'b'])
    
    # Transform data
    transformed = pipeline.transform(df)
    
    # Check log was applied to all columns
    np.testing.assert_allclose(
        transformed['c'].values,
        np.log10(df['c'].values)
    )
    
    # Check minmax was applied only to a and b
    for i in range(len(df)):
        row_subset = transformed.iloc[i][['a', 'b']]
        assert np.isclose(row_subset.min(), 0.0) or np.isclose(row_subset.max(), 1.0)
    
    # Test inverse transform
    back_transformed = pipeline.inverse_transform(transformed)
    
    # Check we get back close to original values
    np.testing.assert_allclose(back_transformed.values, df.values, rtol=1e-5)

def test_autobots_assemble():
    """Test the AutobotsAssemble class functionality"""
    # Create test data
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [10, 20, 30, 40],
        'c': [-10, -20, -30, -40]
    })
    
    # Save original data for comparison
    original_df = df.copy()
    
    # Initialize with data
    aa = pyemu.emulators.AutobotsAssemble(df)
    
    # Apply log transform to positive columns
    aa.apply('log10', columns=['a', 'b'])
    
    # Check the transform was applied correctly
    np.testing.assert_allclose(
        aa.df[['a', 'b']].values,
        np.log10(original_df[['a', 'b']].values)
    )
    
    # Check that column c is unchanged
    np.testing.assert_array_equal(aa.df['c'].values, original_df['c'].values)
    
    # Save intermediate state after log transform
    log_transformed = aa.df.copy()
    
    # Apply normal score transform to all columns
    aa.apply('normal_score')
    
    # Save state after normal score transform
    normal_transformed = aa.df.copy()
    
    # Verify both transforms were applied (data should be different from log transform)
    assert not np.allclose(normal_transformed.values, log_transformed.values)
    
    # Apply the inverse transformation
    back_transformed = aa.inverse()
    
    # Check we get back close to original values
    np.testing.assert_allclose(back_transformed.values, original_df.values, rtol=0.1)
    
    # Test with external already-transformed data
    external_transformed = pd.DataFrame({
        'a': [-0.5, 0.0, 0.5],  # Already transformed data in normal score space
        'b': [0.5, 0.0, -0.5],  # (approximately in the normal distribution range)
        'c': [1.0, 0.0, -1.0]
    })
    
    # Test inverse transform on external transformed data
    back_external = aa.inverse(external_transformed)
    
    # Check that shape is preserved
    assert back_external.shape == external_transformed.shape
    
    # Verify output has reasonable values (should be in the range of original data)
    for col in ['a', 'b']:
        # These columns had log transform applied, so should be positive
        assert np.all(back_external[col] > 0)
    
    # Column c should have values in the range of the original data
    assert np.min(back_external['c']) >= -40
    assert np.max(back_external['c']) <= -10
    
    # Apply transform again to verify roundtrip accuracy
    roundtrip = aa.transform(back_external)
    
    # Check roundtrip accuracy for values within standard normal range (-2 to 2)
    for col in external_transformed.columns:
        # Find values within the normal range
        mask = (external_transformed[col] >= -2) & (external_transformed[col] <= 2)
        if mask.any():
            # Get the values to compare
            expected = external_transformed.loc[mask, col].values
            actual = roundtrip.loc[mask, col].values
            
            # Handle zeros and near-zeros with absolute tolerance instead of relative
            zero_mask = np.isclose(expected, 0, atol=1e-10)
            if zero_mask.any():
                # For zeros, use absolute tolerance
                np.testing.assert_allclose(
                    actual[zero_mask],
                    expected[zero_mask],
                    atol=0.1  # Absolute tolerance for zeros
                )
                
                # For non-zeros, use relative tolerance
                if (~zero_mask).any():
                    np.testing.assert_allclose(
                        actual[~zero_mask],
                        expected[~zero_mask],
                        rtol=0.1  # Relative tolerance for non-zeros
                    )
            else:
                # No zeros, use normal comparison
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=0.1
                )
    
    # Additional test to verify pipeline order is maintained
    # Create a new pipeline with transforms in different order
    bb = pyemu.emulators.AutobotsAssemble(original_df.copy())
    
    # First normal score, then log10
    bb.apply('normal_score')
    bb.apply('log10', columns=['a', 'b'])
    
    # Apply inverse - should revert log10 first, then normal_score
    back_bb = bb.inverse()
    
    # Check we get back close to original values
    np.testing.assert_allclose(back_bb.values, original_df.values, rtol=0.1)



def test_normal_score_tied_minimum_strictly_increasing():
    """Regression: ties at the data minimum (magnitude >= 1) must not survive
    in the fitted 'originals' array, which must be strictly increasing."""
    # Detection-limit style run: 8 ties at 100.0 then an increasing tail
    col = 'tied_min'
    values = np.concatenate([np.full(8, 100.0), np.linspace(101, 120, 22)])
    df = pd.DataFrame({col: values})

    nst = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=False)
    nst.fit(df)

    originals = np.asarray(nst.column_parameters[col]['originals'])

    # Tied values must have been separated so np.interp gets a valid xp
    assert np.all(np.diff(originals) > 0)


def test_normal_score_tied_minimum_extrapolation_finite():
    """Regression: with tied minimum values and quadratic extrapolation,
    transforming a below-minimum value must be finite (not +/-inf)."""
    col = 'tied_min'
    values = np.concatenate([np.full(8, 100.0), np.linspace(101, 120, 22)])
    df = pd.DataFrame({col: values})

    nst = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
    nst.fit(df)

    # 90.0 is below the data minimum (100.0), forcing below-min extrapolation
    probe = pd.DataFrame({col: [90.0]})
    transformed = nst.transform(probe)

    # Only finiteness is asserted; magnitude of degenerate-pair
    # extrapolation is intentionally out of scope
    assert np.all(np.isfinite(transformed[col].values))


def test_normal_score_tied_maximum_extrapolation_finite():
    """Regression: with tied maximum values and quadratic extrapolation,
    transforming an above-maximum value must be finite (not +/-inf)."""
    col = 'tied_max'
    values = np.concatenate([np.linspace(80, 99, 22), np.full(8, 100.0)])
    df = pd.DataFrame({col: values})

    nst = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
    nst.fit(df)

    # 110.0 is above the data maximum (100.0), forcing above-max extrapolation
    probe = pd.DataFrame({col: [110.0]})
    transformed = nst.transform(probe)

    assert np.all(np.isfinite(transformed[col].values))


def test_normal_score_tied_minimum_round_trip():
    """Sanity: round-trip inverse_transform(transform(x)) recovers in-range
    probes on the tied-minimum data."""
    col = 'tied_min'
    values = np.concatenate([np.full(8, 100.0), np.linspace(101, 120, 22)])
    df = pd.DataFrame({col: values})

    nst = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
    nst.fit(df)

    probe = pd.DataFrame({col: [100.0, 100.5, 110.0]})
    round_trip = nst.inverse_transform(nst.transform(probe))

    np.testing.assert_allclose(round_trip[col].values, probe[col].values)


def skewness(x):
    """Calculate skewness of a distribution"""
    n = len(x)
    x_mean = np.mean(x)
    return (np.sum((x - x_mean) ** 3) / n) / ((np.sum((x - x_mean) ** 2) / n) ** 1.5)

def kurtosis(x):
    """Calculate kurtosis of a distribution"""
    n = len(x)
    x_mean = np.mean(x)
    return (np.sum((x - x_mean) ** 4) / n) / ((np.sum((x - x_mean) ** 2) / n) ** 2)




def test_normal_score_with_external_data():
    """Test NormalScoreTransformer with external already-transformed data"""
    # Create training data with a specific distribution
    rng = np.random.RandomState(42)
    n = 100
    training_data = pd.DataFrame({
        'normal': rng.normal(5, 2, n),
        'lognormal': np.exp(rng.normal(1, 0.5, n)),
        'uniform': rng.uniform(0, 10, n)
    })
    
    # Create "external" data that we'll pretend is already transformed
    # For this test, we'll generate values in the typical normal score range (-3 to 3)
    external_transformed = pd.DataFrame({
        'normal': rng.normal(0, 1, 1),  # Already in normal score space
        'lognormal': rng.normal(0, 1, 1),
        'uniform': rng.normal(0, 1, 1)
    })
    
    # Initialize and fit transformer on training data
    nst = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
    nst.fit(training_data)
    
    # Transform training data to verify transformation works
    transformed_training = nst.transform(training_data)
    
    # Check that transformed data has properties of normal distribution
    for col in training_data.columns:
        # Mean should be close to 0
        assert abs(transformed_training[col].mean()) < 0.3
        # Standard deviation should be close to 1
        assert abs(transformed_training[col].std() - 1.0) < 0.3
    
    # Store column parameters for inspection
    z_scores = {}
    originals = {}
    for col in training_data.columns:
        params = nst.column_parameters.get(col, {})
        z_scores[col] = params.get('z_scores', [])
        originals[col] = params.get('originals', [])
        
        # Verify column parameters were created
        assert len(z_scores[col]) > 0
        assert len(originals[col]) > 0
    
    # Apply inverse transform to external transformed data directly
    back_external = nst.inverse_transform(external_transformed)
    
    # Verify the shape matches
    assert back_external.shape == external_transformed.shape
    
    # Apply the transform to back_external to check if it recovers external_transformed
    re_transformed = nst.transform(back_external)
    
    # Check that re-transforming recovers values close to the external_transformed
    # Note: exact recovery isn't expected due to interpolation/extrapolation
    for col in external_transformed.columns:
        # Values inside the normal range (-2 to 2) should be very close
        inside_range = (external_transformed[col] >= -2) & (external_transformed[col] <= 2)
        if inside_range.any():
            np.testing.assert_allclose(
                re_transformed.loc[inside_range, col].values,
                external_transformed.loc[inside_range, col].values,
                rtol=0.2
            )
    
    # Test external values that are far outside the z-score range
    extreme_transformed = pd.DataFrame({
        'normal': np.array([-5, 0, 5],dtype=float),  # Includes extreme values
        'lognormal': np.array([-5, 0, 5],dtype=float),
        'uniform': np.array([-5, 0, 5],dtype=float)
    })
    
    # Test with extrapolation first
    nst_extrap = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
    nst_extrap.fit(training_data)
    back_extreme_extrap = nst_extrap.inverse_transform(extreme_transformed)
    
    # Test without extrapolation
    nst_no_extrap = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=False)
    nst_no_extrap.fit(training_data)
    back_extreme_no_extrap = nst_no_extrap.inverse_transform(extreme_transformed)
    
    # With extrapolation, extreme values should be outside the original data range
    for col in training_data.columns:
        min_orig = training_data[col].min()
        max_orig = training_data[col].max()
        
        # Check extrapolation is working (values outside original range)
        assert back_extreme_extrap[col].min() < min_orig or back_extreme_extrap[col].max() > max_orig
        
        # Without extrapolation, values should be clamped to original range
        assert back_extreme_no_extrap[col].min() >= min_orig - 1e-10  # Allow for floating point error
        assert back_extreme_no_extrap[col].max() <= max_orig + 1e-10
    
    # Test with AutobotsAssemble to ensure the pipeline works with external transformed data
    aa = pyemu.emulators.AutobotsAssemble(training_data.copy())
    aa.apply('normal_score')
    
    # Test applying inverse transform to external data
    back_from_aa = aa.inverse(external_transformed.copy())
    
    # Verify results with direct inverse transform
    np.testing.assert_allclose(
        back_from_aa.values,
        nst.inverse_transform(external_transformed).values,
        rtol=1e-3
    )


def _blom_reference(n):
    """Analytical Blom expected normal order statistics, computed independently."""
    from scipy.stats import norm
    i = np.arange(1, n + 1)
    return norm.ppf((i - 0.375) / (n + 0.25))


def _rng_states_equal(s1, s2):
    """Compare two pyemu.en.rng get_state() tuples for exact equality."""
    # RandomState.get_state() -> (str, ndarray[uint32], int, int, float)
    if s1[0] != s2[0]:
        return False
    if not np.array_equal(s1[1], s2[1]):
        return False
    return s1[2:] == s2[2:]


def test_normal_score_fit_consumes_no_global_rng():
    """fit must be deterministic and draw no numbers from the module-global RNG."""
    rng = np.random.RandomState(7)
    df = pd.DataFrame({
        'a': rng.normal(0, 1, 40),
        'b': rng.uniform(0, 5, 40),
    })

    before = pyemu.en.rng.get_state()
    nst = pyemu.emulators.NormalScoreTransformer()
    nst.fit(df)
    after = pyemu.en.rng.get_state()

    assert _rng_states_equal(before, after), \
        "NormalScoreTransformer.fit consumed numbers from pyemu.en.rng"


def test_normal_score_fit_is_deterministic_across_rng_advance():
    """Two separate fits on the same frame must be identical even when the
    global RNG is advanced between them (fit consumes no random numbers)."""
    rng = np.random.RandomState(11)
    df = pd.DataFrame({
        'a': rng.normal(0, 1, 40),
        'b': rng.uniform(0, 5, 40),
    })

    nst1 = pyemu.emulators.NormalScoreTransformer()
    nst1.fit(df)

    # advance the global RNG between the two fits
    pyemu.en.rng.normal(size=1000)

    nst2 = pyemu.emulators.NormalScoreTransformer()
    nst2.fit(df)

    for col in df.columns:
        np.testing.assert_array_equal(
            nst1.column_parameters[col]['z_scores'],
            nst2.column_parameters[col]['z_scores'],
        )

    out1 = nst1.transform(df)
    out2 = nst2.transform(df)
    np.testing.assert_array_equal(out1.values, out2.values)


def test_normal_score_z_table_matches_blom():
    """Fitted z_scores must equal Blom's analytical formula exactly, be strictly
    increasing, and be antisymmetric."""
    n = 40
    rng = np.random.RandomState(13)
    df = pd.DataFrame({'a': rng.normal(0, 1, n)})

    nst = pyemu.emulators.NormalScoreTransformer()
    nst.fit(df)

    z_scores = np.asarray(nst.column_parameters['a']['z_scores'])
    expected = _blom_reference(n)

    # exact equality with the analytical Blom formula
    np.testing.assert_array_equal(z_scores, expected)

    # strictly increasing
    assert np.all(np.diff(z_scores) > 0)

    # antisymmetric: z == -z[::-1]
    np.testing.assert_allclose(z_scores, -z_scores[::-1])


def test_normal_score_deprecated_kwargs_are_noops():
    """tol and max_samples are accepted for backward compat and have no effect
    on the fitted z-table."""
    rng = np.random.RandomState(17)
    df = pd.DataFrame({'a': rng.normal(0, 1, 40)})

    nst_default = pyemu.emulators.NormalScoreTransformer()
    nst_default.fit(df)

    nst_kwargs = pyemu.emulators.NormalScoreTransformer(tol=1e-9, max_samples=5)
    nst_kwargs.fit(df)

    np.testing.assert_array_equal(
        nst_default.column_parameters['a']['z_scores'],
        nst_kwargs.column_parameters['a']['z_scores'],
    )


# ---------------------------------------------------------------------------
# Quadratic-extrapolation tail tests (monotone quadratic tails replacing the
# former linear end-pair extrapolation in NormalScoreTransformer).
# ---------------------------------------------------------------------------

def _curved_frame(col='c', n=60, seed=42):
    """Curved single-column frame: np.exp of seeded normals (genuinely curved
    in original space, so the quadratic tails differ from a straight line)."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({col: np.exp(rng.normal(0, 1, n))})


def _old_linear_below_z(originals, z_scores, v):
    """OLD (pre-fix) forward extrapolation below the data minimum: linear in the
    end knot PAIR, slope = (z1 - z0) / (o1 - o0)."""
    o = np.asarray(originals, dtype=float)
    z = np.asarray(z_scores, dtype=float)
    min_orig, min_z = o.min(), z.min()
    slope = (z[1] - z[0]) / (o[1] - o[0])
    return min_z + slope * (v - min_orig)


def _old_linear_above_z(originals, z_scores, v):
    """OLD (pre-fix) forward extrapolation above the data maximum: linear in the
    end knot PAIR, slope = (z[-1] - z[-2]) / (o[-1] - o[-2])."""
    o = np.asarray(originals, dtype=float)
    z = np.asarray(z_scores, dtype=float)
    max_orig, max_z = o.max(), z.max()
    slope = (z[-1] - z[-2]) / (o[-1] - o[-2])
    return max_z + slope * (v - max_orig)


def test_normal_score_quadratic_tail_round_trip_both_tails():
    """(a) Round-trip exactness in BOTH tails on curved data:
    inverse_transform(transform(x)) == x for probes below the data minimum and
    above the data maximum (transform uses the cancellation-free quadratic root
    so it is an exact inverse of the tail curve)."""
    col = 'c'
    df = _curved_frame(col)

    nst = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
    nst.fit(df)

    originals = np.asarray(nst.column_parameters[col]['originals'])
    min_orig, max_orig = originals.min(), originals.max()
    span = max_orig - min_orig

    # probes strictly below the minimum and strictly above the maximum
    below = np.array([min_orig - 0.05 * span, min_orig - 0.5 * span, min_orig - span])
    above = np.array([max_orig + 0.05 * span, max_orig + 0.5 * span, max_orig + span])
    probe = pd.DataFrame({col: np.concatenate([below, above])})

    round_trip = nst.inverse_transform(nst.transform(probe))
    np.testing.assert_allclose(round_trip[col].values, probe[col].values, rtol=1e-9, atol=1e-9)


def test_normal_score_quadratic_tail_is_actually_quadratic():
    """(b) The upper tail honours curvature: inverse over a uniform z-grid beyond
    the max z has a nonzero second difference (a straight line would give zero),
    AND the forward transform of an out-of-range value differs from the OLD
    end-pair linear extrapolation formula."""
    col = 'c'
    df = _curved_frame(col)

    nst = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
    nst.fit(df)

    originals = np.asarray(nst.column_parameters[col]['originals'])
    z_scores = np.asarray(nst.column_parameters[col]['z_scores'])
    max_z = z_scores.max()
    max_orig = originals.max()

    # uniform z-grid beyond the max z; inverse should be curved (nonzero d2)
    z_grid = max_z + np.linspace(0.2, 2.0, 9)
    inv = nst.inverse_transform(pd.DataFrame({col: z_grid}))[col].values
    second_diff = np.diff(inv, 2)
    assert np.max(np.abs(second_diff)) > 1e-8, "upper tail is straight, not quadratic"

    # forward transform of an above-max value must differ from OLD linear formula
    span = max_orig - originals.min()
    v = max_orig + 0.5 * span
    got = nst.transform(pd.DataFrame({col: [v]}))[col].values[0]
    old_linear = _old_linear_above_z(originals, z_scores, v)
    assert not np.isclose(got, old_linear, rtol=1e-6, atol=1e-9), \
        "quadratic forward transform coincides with old linear end-pair formula"


def test_normal_score_quadratic_tail_monotone_and_finite():
    """(c) transform of a fine grid spanning [min-2, max+2] is strictly
    increasing and finite, for BOTH the curved data and a tied-end dataset."""
    curved = _curved_frame('c')
    tied = pd.DataFrame({'t': np.concatenate([np.linspace(1, 99, 32), np.full(8, 100.0)])})

    for df, col in [(curved, 'c'), (tied, 't')]:
        nst = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
        nst.fit(df)

        originals = np.asarray(nst.column_parameters[col]['originals'])
        grid = np.linspace(originals.min() - 2.0, originals.max() + 2.0, 500)
        z = nst.transform(pd.DataFrame({col: grid}))[col].values

        assert np.all(np.isfinite(z)), f"non-finite transform on {col}"
        assert np.all(np.diff(z) > 0), f"transform not strictly increasing on {col}"


def test_normal_score_tied_end_residual_regression():
    """(d) Tied-end sanity (f9-residual regression): on the tied-end dataset,
    transform(100.5) is a small z-score between 2 and 10 (pre-fix the degenerate
    end-pair slope blew this up to ~1e13)."""
    col = 't'
    df = pd.DataFrame({col: np.concatenate([np.linspace(1, 99, 32), np.full(8, 100.0)])})

    nst = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
    nst.fit(df)

    z = nst.transform(pd.DataFrame({col: [100.5]}))[col].values[0]
    assert 2.0 < z < 10.0, f"tied-end transform(100.5) = {z}, expected in (2, 10)"


def test_normal_score_quadratic_n1_clamps_no_crash():
    """(e) n=1 with quadratic_extrapolation=True: no crash (pre-fix this raised
    IndexError); out-of-range values clamp to the boundary."""
    col = 'c'
    df = pd.DataFrame({col: [5.0]})

    nst = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
    nst.fit(df)

    originals = np.asarray(nst.column_parameters[col]['originals'])
    z_scores = np.asarray(nst.column_parameters[col]['z_scores'])
    min_z, max_z = z_scores.min(), z_scores.max()
    min_orig, max_orig = originals.min(), originals.max()

    # forward transform of below/above values must not crash and must clamp to z
    fwd = nst.transform(pd.DataFrame({col: [1.0, 5.0, 9.0]}))[col].values
    assert np.all(np.isfinite(fwd))
    assert fwd[0] == min_z
    assert fwd[2] == max_z

    # inverse transform of below/above z must clamp to the boundary original
    inv = nst.inverse_transform(pd.DataFrame({col: [min_z - 3.0, max_z + 3.0]}))[col].values
    assert np.all(np.isfinite(inv))
    assert inv[0] == min_orig
    assert inv[1] == max_orig


def test_normal_score_quadratic_constant_column_clamps():
    """(f) Constant column with quadratic_extrapolation=True: out-of-range values
    clamp to the boundary z-scores / originals (no explosion)."""
    col = 'c'
    df = pd.DataFrame({col: np.full(40, 7.0)})

    nst = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
    nst.fit(df)

    originals = np.asarray(nst.column_parameters[col]['originals'])
    z_scores = np.asarray(nst.column_parameters[col]['z_scores'])
    min_z, max_z = z_scores.min(), z_scores.max()
    min_orig, max_orig = originals.min(), originals.max()

    fwd = nst.transform(pd.DataFrame({col: [1.0, 100.0]}))[col].values
    assert np.all(np.isfinite(fwd))
    assert fwd[0] == min_z
    assert fwd[1] == max_z

    inv = nst.inverse_transform(pd.DataFrame({col: [min_z - 5.0, max_z + 5.0]}))[col].values
    assert np.all(np.isfinite(inv))
    assert inv[0] == min_orig
    assert inv[1] == max_orig


def test_normal_score_clamp_false_unchanged():
    """(g) clamp behaviour (quadratic_extrapolation=False) is unchanged:
    out-of-range maps exactly to the boundary z / boundary original."""
    col = 'c'
    df = _curved_frame(col)

    nst = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=False)
    nst.fit(df)

    originals = np.asarray(nst.column_parameters[col]['originals'])
    z_scores = np.asarray(nst.column_parameters[col]['z_scores'])
    min_z, max_z = z_scores.min(), z_scores.max()
    min_orig, max_orig = originals.min(), originals.max()
    span = max_orig - min_orig

    # forward: below/above the data range clamp exactly to boundary z
    fwd = nst.transform(pd.DataFrame({col: [min_orig - span, max_orig + span]}))[col].values
    assert fwd[0] == min_z
    assert fwd[1] == max_z

    # inverse: below/above the z range clamp exactly to boundary original
    inv = nst.inverse_transform(pd.DataFrame({col: [min_z - 5.0, max_z + 5.0]}))[col].values
    assert inv[0] == min_orig
    assert inv[1] == max_orig


# ---------------------------------------------------------------------------
# Linear-extrapolation tail tests, the extrapolation/quadratic_extrapolation
# alias mapping, invalid-value handling, config wiring, and pre-feature pickle
# compatibility for the new ``extrapolation`` parameter.
# ---------------------------------------------------------------------------

def test_normal_score_linear_tail_straight_roundtrip_and_finite():
    """(a) linear mode: each tail is a straight line (zero second difference of
    the inverse on a z-grid beyond the max), round-trips exactly in BOTH tails,
    and the forward transform across the boundary is monotone and finite."""
    col = 'c'
    df = _curved_frame(col)

    nst = pyemu.emulators.NormalScoreTransformer(extrapolation="linear")
    nst.fit(df)
    assert nst.extrapolation == "linear"

    originals = np.asarray(nst.column_parameters[col]['originals'])
    z_scores = np.asarray(nst.column_parameters[col]['z_scores'])
    min_orig, max_orig = originals.min(), originals.max()
    max_z = z_scores.max()
    span = max_orig - min_orig

    # inverse over a uniform z-grid beyond the max z must be straight: the
    # second difference of a line is zero (contrast the quadratic tail test).
    z_grid = max_z + np.linspace(0.2, 2.0, 9)
    inv = nst.inverse_transform(pd.DataFrame({col: z_grid}))[col].values
    second_diff = np.diff(inv, 2)
    np.testing.assert_allclose(second_diff, 0.0, atol=1e-8)

    # exact round-trip in both tails
    below = np.array([min_orig - 0.05 * span, min_orig - 0.5 * span, min_orig - span])
    above = np.array([max_orig + 0.05 * span, max_orig + 0.5 * span, max_orig + span])
    probe = pd.DataFrame({col: np.concatenate([below, above])})
    round_trip = nst.inverse_transform(nst.transform(probe))
    np.testing.assert_allclose(round_trip[col].values, probe[col].values, rtol=1e-9, atol=1e-9)

    # monotone and finite across the boundary on a fine grid
    grid = np.linspace(min_orig - 2.0, max_orig + 2.0, 500)
    z = nst.transform(pd.DataFrame({col: grid}))[col].values
    assert np.all(np.isfinite(z))
    assert np.all(np.diff(z) > 0)


def test_normal_score_linear_tail_tie_robust():
    """(a) linear mode is tie-robust: on the tied-end dataset
    linspace(1, 99, 32) + 8x100.0 the adaptive knot selection skips the tied
    knots, so transform(100.5) is a small z in (2, 10) (a degenerate end-pair
    slope would blow this up)."""
    col = 't'
    df = pd.DataFrame({col: np.concatenate([np.linspace(1, 99, 32), np.full(8, 100.0)])})

    nst = pyemu.emulators.NormalScoreTransformer(extrapolation="linear")
    nst.fit(df)

    z = nst.transform(pd.DataFrame({col: [100.5]}))[col].values[0]
    assert 2.0 < z < 10.0, f"tied-end linear transform(100.5) = {z}, expected in (2, 10)"


def test_normal_score_extrapolation_alias_mapping():
    """(b) the deprecated boolean maps onto the resolved .extrapolation, and an
    explicit extrapolation overrides it:
      - quadratic_extrapolation=True  -> .extrapolation == "quadratic", curved tails
      - default                       -> .extrapolation == "clamp", out-of-range clamps
      - extrapolation="clamp" + quadratic_extrapolation=True -> "clamp" wins
    """
    col = 'c'
    df = _curved_frame(col)

    # quadratic_extrapolation=True maps to "quadratic" with curved tails
    nst_quad = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
    nst_quad.fit(df)
    assert nst_quad.extrapolation == "quadratic"
    z_scores = np.asarray(nst_quad.column_parameters[col]['z_scores'])
    max_z = z_scores.max()
    z_grid = max_z + np.linspace(0.2, 2.0, 9)
    inv = nst_quad.inverse_transform(pd.DataFrame({col: z_grid}))[col].values
    assert np.max(np.abs(np.diff(inv, 2))) > 1e-8, "quadratic alias did not produce curved tails"

    # default maps to "clamp" and out-of-range values clamp to the boundary
    nst_def = pyemu.emulators.NormalScoreTransformer()
    nst_def.fit(df)
    assert nst_def.extrapolation == "clamp"
    originals = np.asarray(nst_def.column_parameters[col]['originals'])
    min_orig, max_orig = originals.min(), originals.max()
    zs = np.asarray(nst_def.column_parameters[col]['z_scores'])
    min_z, max_z = zs.min(), zs.max()
    span = max_orig - min_orig
    fwd = nst_def.transform(pd.DataFrame({col: [min_orig - span, max_orig + span]}))[col].values
    assert fwd[0] == min_z
    assert fwd[1] == max_z

    # explicit extrapolation wins over the deprecated boolean
    nst_override = pyemu.emulators.NormalScoreTransformer(
        extrapolation="clamp", quadratic_extrapolation=True)
    assert nst_override.extrapolation == "clamp"


def test_normal_score_invalid_extrapolation_raises():
    """(c) an invalid extrapolation value raises ValueError at construction."""
    with pytest.raises(ValueError):
        pyemu.emulators.NormalScoreTransformer(extrapolation="cubic")


def test_normal_score_autobots_apply_passes_extrapolation():
    """(d) config wiring: AutobotsAssemble(df).apply("normal_score",
    extrapolation="linear") stores a NormalScoreTransformer whose
    .extrapolation == "linear"."""
    col = 'c'
    df = _curved_frame(col)

    aa = pyemu.emulators.AutobotsAssemble(df)
    aa.apply("normal_score", extrapolation="linear")

    transformer = aa.pipeline.transformers[0][0]
    assert isinstance(transformer, pyemu.emulators.NormalScoreTransformer)
    assert transformer.extrapolation == "linear"


def test_normal_score_pre_feature_pickle_compat():
    """(e) pre-feature pickle compat: an instance lacking the .extrapolation
    attribute (as unpickled from before the feature existed) falls back to the
    deprecated boolean. After deleting .extrapolation, transforming an
    out-of-range value still works (finite) and _extrapolation_mode() resolves
    to "quadratic"."""
    col = 'c'
    df = _curved_frame(col)

    t = pyemu.emulators.NormalScoreTransformer(quadratic_extrapolation=True)
    t.fit(df)

    # simulate a pre-feature pickle: the attribute simply isn't there
    del t.extrapolation

    assert t._extrapolation_mode() == "quadratic"

    originals = np.asarray(t.column_parameters[col]['originals'])
    max_orig = originals.max()
    span = max_orig - originals.min()
    out = t.transform(pd.DataFrame({col: [max_orig + 0.5 * span]}))[col].values
    assert np.all(np.isfinite(out))