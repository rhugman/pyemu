"""
Unit tests for pyemu.emulators — no PEST++ binaries required.

Priority 1: Core logic (DSI, GPR)
Priority 2: Transformer unit tests
Priority 3: Base class / file generation
"""
import os
import warnings
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# DSIAE-backed tests need TensorFlow; gate them so the file still collects without it.
try:
    import tensorflow as tf  # noqa: F401
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synth_data(n_real=100, n_obs=10, seed=42):
    """Generate synthetic ensemble data and dummy observation metadata."""
    np.random.seed(seed)
    data = pd.DataFrame(
        np.random.normal(size=(n_real, n_obs)),
        columns=[f"obs{i}" for i in range(n_obs)],
    )
    obsdata = pd.DataFrame(
        {
            "obsnme": data.columns,
            "obsval": data.mean().values,
            "weight": 1.0,
            "obgnme": "obgnme",
        },
        index=data.columns,
    )
    return data, obsdata


# ===========================================================================
# PRIORITY 1 — Core unit tests (DSI & GPR)
# ===========================================================================


class TestDSIFitPredict:
    """Unit-level tests for DSI core logic (no PEST++ binaries)."""

    def test_fit_predict_basic(self):
        """DSI fit+predict with no transforms returns correct shape/type."""
        from pyemu.emulators import DSI

        data, obsdata = _synth_data()
        dsi = DSI(data=data, pst=obsdata, verbose=False)
        dsi.fit()

        assert dsi.fitted
        assert dsi.pmat is not None
        assert dsi.ovals is not None

        # Predict with zero pvals — should recover the mean (ovals)
        pvals = np.zeros(dsi.pmat.shape[1])
        result = dsi.predict(pvals)
        assert isinstance(result, pd.Series)
        assert len(result) == data.shape[1]
        np.testing.assert_allclose(result.values, dsi.ovals.values, atol=1e-10)

    def test_predict_multi_realization(self):
        """DSI.predict with DataFrame input returns DataFrame with correct shape."""
        from pyemu.emulators import DSI

        data, obsdata = _synth_data()
        dsi = DSI(data=data, pst=obsdata, verbose=False)
        dsi.fit()

        n_reals = 5
        pvals_df = pd.DataFrame(
            np.random.normal(size=(n_reals, dsi.pmat.shape[1])),
            columns=[f"p_{i}" for i in range(dsi.pmat.shape[1])],
        )
        result = dsi.predict(pvals_df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (n_reals, data.shape[1])

    def test_predict_series_vs_array(self):
        """Predictions from Series, 1-D array, and 2-D array are identical."""
        from pyemu.emulators import DSI

        data, obsdata = _synth_data()
        dsi = DSI(data=data, pst=obsdata, verbose=False)
        dsi.fit()

        n_pars = dsi.pmat.shape[1]
        vals = np.random.normal(size=n_pars)

        result_array_1d = dsi.predict(vals)
        result_array_2d = dsi.predict(vals.reshape(1, -1))
        result_series = dsi.predict(pd.Series(vals, index=[f"p_{i}" for i in range(n_pars)]))

        np.testing.assert_allclose(result_array_1d.values, result_array_2d.values.flatten(), atol=1e-12)
        np.testing.assert_allclose(result_array_1d.values, result_series.values, atol=1e-12)

    def test_energy_truncation(self):
        """SVD truncation at energy_threshold < 1.0 reduces dimensionality."""
        from pyemu.emulators import DSI

        data, obsdata = _synth_data(n_real=100, n_obs=10)
        dsi_full = DSI(data=data, pst=obsdata, energy_threshold=1.0, verbose=False)
        dsi_full.fit()

        dsi_trunc = DSI(data=data, pst=obsdata, energy_threshold=0.8, verbose=False)
        dsi_trunc.fit()

        # Truncated should have fewer (or equal) parameters
        assert dsi_trunc.pmat.shape[1] <= dsi_full.pmat.shape[1]
        # Energy captured should be >= threshold
        s = dsi_trunc.s
        energy = np.sum(s ** 2) / np.sum(dsi_full.s ** 2)
        assert energy >= 0.8

    def test_predict_rowwise_inverse(self):
        """Row-wise scaling + inverse returns values in physical space."""
        from pyemu.emulators import DSI

        data, obsdata = _synth_data()
        rowwise_groups = {
            "g1": [f"obs{i}" for i in range(5)],
            "g2": [f"obs{i}" for i in range(5, 10)],
        }
        dsi = DSI(data=data, pst=obsdata, rowwise_groups=rowwise_groups, verbose=False)
        dsi.fit()

        pvals = np.zeros(dsi.pmat.shape[1])
        result = dsi.predict(pvals)
        assert isinstance(result, pd.Series)
        assert len(result) == data.shape[1]
        # Values should be finite (no NaN / Inf from inverse scaling)
        assert np.all(np.isfinite(result.values))

    def test_save_load(self, tmp_path):
        """Round-trip pickle preserves predictions."""
        from pyemu.emulators import DSI

        data, obsdata = _synth_data()
        dsi = DSI(data=data, pst=obsdata, verbose=False)
        dsi.fit()

        pvals = np.random.normal(size=dsi.pmat.shape[1])
        pred_before = dsi.predict(pvals)

        path = str(tmp_path / "dsi_test.pkl")
        dsi.save(path)
        dsi_loaded = DSI.load(path)

        pred_after = dsi_loaded.predict(pvals)
        np.testing.assert_allclose(pred_before.values, pred_after.values, atol=1e-12)

    def test_predict_before_fit_raises(self):
        """Predicting before fit raises ValueError."""
        from pyemu.emulators import DSI

        data, obsdata = _synth_data()
        dsi = DSI(data=data, pst=obsdata, verbose=False)
        # dsi is NOT fitted
        with pytest.raises(ValueError, match="fitted"):
            dsi.predict(np.zeros(5))

    def test_predict_dimension_mismatch_raises(self):
        """Wrong pvals dimension raises ValueError."""
        from pyemu.emulators import DSI

        data, obsdata = _synth_data()
        dsi = DSI(data=data, pst=obsdata, verbose=False)
        dsi.fit()

        wrong_size = dsi.pmat.shape[1] + 3
        with pytest.raises(ValueError, match="parameters"):
            dsi.predict(np.zeros(wrong_size))

    def test_fit_with_transforms(self):
        """DSI with normal_score transform fits and round-trips."""
        from pyemu.emulators import DSI

        data, obsdata = _synth_data()
        transforms = [{"type": "normal_score"}]
        dsi = DSI(data=data, pst=obsdata, transforms=transforms, verbose=False)
        dsi.fit()

        assert dsi.fitted
        pvals = np.zeros(dsi.pmat.shape[1])
        result = dsi.predict(pvals)
        assert len(result) == data.shape[1]
        assert np.all(np.isfinite(result.values))

    def test_fit_with_mixed_transforms(self):
        """DSI with log10 + normal_score fits and round-trips."""
        from pyemu.emulators import DSI

        data, _ = _synth_data()
        # Make first 2 columns positive for log10
        data["obs0"] = np.abs(data["obs0"]) + 1.0
        data["obs1"] = np.abs(data["obs1"]) + 1.0
        obsdata = pd.DataFrame(
            {"obsnme": data.columns, "obsval": data.mean().values, "weight": 1.0, "obgnme": "obgnme"},
            index=data.columns,
        )

        transforms = [
            {"type": "log10", "columns": ["obs0", "obs1"]},
            {"type": "normal_score"},
        ]
        dsi = DSI(data=data, pst=obsdata, transforms=transforms, verbose=False)
        dsi.fit()

        pvals = np.zeros(dsi.pmat.shape[1])
        result = dsi.predict(pvals)
        assert len(result) == data.shape[1]
        assert np.all(np.isfinite(result.values))


class TestGPRFitPredict:
    """Unit-level tests for GPR core logic (no PEST++ binaries)."""

    @staticmethod
    def _simple_gpr_data():
        """y = 2*x + 1, perfect linear."""
        x = np.linspace(0.0, 10.0, 20)
        y = 2.0 * x + 1.0
        df = pd.DataFrame({"x": x, "y": y})
        return df

    def test_fit_predict_basic(self):
        """GPR fits and predicts a simple linear relationship."""
        from pyemu.emulators import GPR

        df = self._simple_gpr_data()
        gpr = GPR(data=df, input_names=["x"], output_names=["y"], verbose=False)
        gpr.fit()

        assert gpr.fitted
        pred = gpr.predict(df[["x"]])
        diff = np.abs(pred["y"].values - df["y"].values)
        assert np.max(diff) < 0.1

    def test_predict_return_std(self):
        """return_std=True returns tuple of (predictions, std)."""
        from pyemu.emulators import GPR

        df = self._simple_gpr_data()
        gpr = GPR(data=df, input_names=["x"], output_names=["y"], verbose=False)
        gpr.fit()

        result = gpr.predict(df[["x"]], return_std=True)
        assert isinstance(result, tuple)
        preds, stds = result
        assert isinstance(preds, pd.DataFrame)
        assert isinstance(stds, pd.DataFrame)
        assert stds.shape == preds.shape
        # Standard deviations should be non-negative
        assert (stds.values >= 0).all()

    def test_custom_kernel(self):
        """GPR with explicit RBF kernel fits successfully."""
        from pyemu.emulators import GPR
        from sklearn.gaussian_process.kernels import RBF

        df = self._simple_gpr_data()
        gpr = GPR(
            data=df,
            input_names=["x"],
            output_names=["y"],
            kernel=RBF(length_scale=1.0),
            verbose=False,
        )
        gpr.fit()
        assert gpr.fitted
        pred = gpr.predict(df[["x"]])
        assert pred.shape == (20, 1)

    def test_multi_output(self):
        """GPR with multiple outputs creates per-output models."""
        from pyemu.emulators import GPR

        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        df = pd.DataFrame({"x": x, "y1": 2 * x + 1, "y2": -x + 5})
        gpr = GPR(data=df, input_names=["x"], output_names=["y1", "y2"], verbose=False)
        gpr.fit()

        assert "y1" in gpr.gpr_models
        assert "y2" in gpr.gpr_models

        pred = gpr.predict(df[["x"]])
        assert pred.shape == (30, 2)
        assert np.max(np.abs(pred["y1"].values - df["y1"].values)) < 0.2
        assert np.max(np.abs(pred["y2"].values - df["y2"].values)) < 0.2

    def test_predict_before_fit_raises(self):
        """Predicting before fit raises ValueError."""
        from pyemu.emulators import GPR

        df = self._simple_gpr_data()
        gpr = GPR(data=df, input_names=["x"], output_names=["y"], verbose=False)
        with pytest.raises(ValueError, match="fitted"):
            gpr.predict(df[["x"]])

    def test_no_transforms(self):
        """GPR with transforms=None works correctly."""
        from pyemu.emulators import GPR

        df = self._simple_gpr_data()
        gpr = GPR(data=df, input_names=["x"], output_names=["y"], transforms=None, verbose=False)
        gpr.fit()
        pred = gpr.predict(df[["x"]])
        assert np.max(np.abs(pred["y"].values - df["y"].values)) < 0.1

    def test_save_load(self, tmp_path):
        """GPR round-trip pickle preserves predictions."""
        from pyemu.emulators import GPR

        df = self._simple_gpr_data()
        gpr = GPR(data=df, input_names=["x"], output_names=["y"], verbose=False)
        gpr.fit()

        pred_before = gpr.predict(df[["x"]])

        path = str(tmp_path / "gpr_test.pkl")
        gpr.save(path)
        gpr_loaded = GPR.load(path)

        pred_after = gpr_loaded.predict(df[["x"]])
        np.testing.assert_allclose(
            pred_before["y"].values, pred_after["y"].values, atol=1e-10
        )


# ===========================================================================
# PRIORITY 2 — Transformer unit tests
# ===========================================================================


class TestNormalScoreTransformer:
    """Tests for NormalScoreTransformer."""

    def test_fit_transform_roundtrip(self):
        """Transform → inverse_transform recovers original values."""
        from pyemu.emulators.transformers import NormalScoreTransformer

        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.exponential(2, 100), "b": np.random.normal(5, 2, 100)})

        nst = NormalScoreTransformer(columns=["a", "b"])
        transformed = nst.fit_transform(df)

        # Transformed values should be approximately normally distributed
        for col in ["a", "b"]:
            vals = transformed[col].values
            assert np.abs(np.mean(vals)) < 0.5  # roughly centered

        inversed = nst.inverse_transform(transformed)
        np.testing.assert_allclose(inversed["a"].values, df["a"].values, atol=0.1)
        np.testing.assert_allclose(inversed["b"].values, df["b"].values, atol=0.1)

    def test_extrapolation(self):
        """Quadratic extrapolation handles values outside training range."""
        from pyemu.emulators.transformers import NormalScoreTransformer

        np.random.seed(42)
        train = pd.DataFrame({"a": np.sort(np.random.normal(0, 1, 50))})
        nst = NormalScoreTransformer(columns=["a"], quadratic_extrapolation=True)
        nst.fit(train)

        # Values outside training range
        test = pd.DataFrame({"a": [train["a"].min() - 2.0, train["a"].max() + 2.0]})
        transformed = nst.transform(test)
        # Should not be clamped — extrapolated values should exceed the fitted z-score bounds
        params = nst.column_parameters["a"]
        min_z, max_z = params["z_scores"].min(), params["z_scores"].max()
        assert transformed["a"].iloc[0] < min_z
        assert transformed["a"].iloc[1] > max_z

        # Inverse round-trip for out-of-range
        inversed = nst.inverse_transform(transformed)
        np.testing.assert_allclose(inversed["a"].values, test["a"].values, atol=0.5)

    def test_no_extrapolation_clamps(self):
        """Without extrapolation, out-of-range values are clamped."""
        from pyemu.emulators.transformers import NormalScoreTransformer

        np.random.seed(42)
        train = pd.DataFrame({"a": np.sort(np.random.normal(0, 1, 50))})
        nst = NormalScoreTransformer(columns=["a"], quadratic_extrapolation=False)
        nst.fit(train)

        test = pd.DataFrame({"a": [train["a"].min() - 5.0, train["a"].max() + 5.0]})
        transformed = nst.transform(test)
        params = nst.column_parameters["a"]
        min_z, max_z = params["z_scores"].min(), params["z_scores"].max()
        assert np.isclose(transformed["a"].iloc[0], min_z)
        assert np.isclose(transformed["a"].iloc[1], max_z)

    def test_selective_columns(self):
        """Only specified columns are transformed."""
        from pyemu.emulators.transformers import NormalScoreTransformer

        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.normal(0, 1, 50), "b": np.random.normal(10, 1, 50)})
        nst = NormalScoreTransformer(columns=["a"])
        transformed = nst.fit_transform(df)
        # Column b should be untouched
        np.testing.assert_array_equal(transformed["b"].values, df["b"].values)


class TestStandardScalerTransformer:
    """Tests for StandardScalerTransformer."""

    def test_fit_transform_stats(self):
        """After transform, mean ≈ 0 and std ≈ 1."""
        from pyemu.emulators.transformers import StandardScalerTransformer

        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.normal(10, 3, 200), "b": np.random.normal(-5, 0.5, 200)})
        sst = StandardScalerTransformer(columns=["a", "b"])
        transformed = sst.fit_transform(df)

        for col in ["a", "b"]:
            assert np.abs(transformed[col].mean()) < 0.05
            assert np.abs(transformed[col].std() - 1.0) < 0.1

    def test_roundtrip(self):
        """Transform → inverse recovers original."""
        from pyemu.emulators.transformers import StandardScalerTransformer

        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.normal(10, 3, 100), "b": np.random.normal(-5, 0.5, 100)})
        sst = StandardScalerTransformer(columns=["a", "b"])
        transformed = sst.fit_transform(df)
        inversed = sst.inverse_transform(transformed)
        np.testing.assert_allclose(inversed.values, df.values, atol=1e-5)

    def test_selective_columns(self):
        """Only specified columns are transformed."""
        from pyemu.emulators.transformers import StandardScalerTransformer

        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.normal(10, 3, 50), "b": [42.0] * 50})
        sst = StandardScalerTransformer(columns=["a"])
        transformed = sst.fit_transform(df)
        np.testing.assert_array_equal(transformed["b"].values, df["b"].values)


class TestMinMaxScalerTransformer:
    """Tests for MinMaxScaler."""

    def test_fit_transform_range(self):
        """Scaled values lie within feature_range."""
        from pyemu.emulators.transformers import MinMaxScaler

        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.uniform(0, 100, 50), "b": np.random.uniform(-10, 10, 50)})
        scaler = MinMaxScaler(feature_range=(0, 1), columns=["a", "b"])
        transformed = scaler.fit_transform(df)
        assert transformed["a"].min() >= -1e-6
        assert transformed["a"].max() <= 1.0 + 1e-6
        assert transformed["b"].min() >= -1e-6
        assert transformed["b"].max() <= 1.0 + 1e-6

    def test_roundtrip(self):
        """Transform → inverse recovers original."""
        from pyemu.emulators.transformers import MinMaxScaler

        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.uniform(0, 100, 50), "b": np.random.uniform(-10, 10, 50)})
        scaler = MinMaxScaler(feature_range=(-1, 1), columns=["a", "b"])
        transformed = scaler.fit_transform(df)
        inversed = scaler.inverse_transform(transformed)
        np.testing.assert_allclose(inversed.values, df.values, atol=1e-5)

    def test_constant_column_skip(self):
        """Constant columns are skipped when skip_constant=True."""
        from pyemu.emulators.transformers import MinMaxScaler

        df = pd.DataFrame({"a": [5.0] * 10, "b": np.arange(10, dtype=float)})
        scaler = MinMaxScaler(feature_range=(0, 1), skip_constant=True)
        transformed = scaler.fit_transform(df)
        # Constant column should be unchanged
        np.testing.assert_array_equal(transformed["a"].values, df["a"].values)
        # Non-constant column should be scaled
        assert transformed["b"].min() >= -1e-6
        assert transformed["b"].max() <= 1.0 + 1e-6

    def test_near_constant_column(self):
        """Near-constant columns (range < 1e-10) are handled gracefully."""
        from pyemu.emulators.transformers import MinMaxScaler

        df = pd.DataFrame({"a": [1.0, 1.0 + 1e-15, 1.0 - 1e-15]})
        scaler = MinMaxScaler(feature_range=(0, 1))
        transformed = scaler.fit_transform(df)
        assert np.all(np.isfinite(transformed.values))


class TestGenericTransformer:
    """Tests for GenericTransformer with sklearn wrappers."""

    def test_power_transformer_roundtrip(self):
        """GenericTransformer wrapping PowerTransformer round-trips."""
        from sklearn.preprocessing import PowerTransformer
        from pyemu.emulators.transformers import GenericTransformer

        np.random.seed(42)
        df = pd.DataFrame({"a": np.abs(np.random.normal(5, 2, 100)), "b": np.abs(np.random.normal(3, 1, 100))})
        gt = GenericTransformer(PowerTransformer, method="yeo-johnson")
        transformed = gt.fit_transform(df)
        inversed = gt.inverse_transform(transformed)
        np.testing.assert_allclose(inversed.values, df.values, atol=1e-4)

    def test_missing_inverse_raises(self):
        """Transformer without inverse_transform raises on init."""
        from pyemu.emulators.transformers import GenericTransformer

        class BadTransformer:
            def fit(self, X):
                return self
            def transform(self, X):
                return X

        with pytest.raises(ValueError, match="inverse_transform"):
            GenericTransformer(BadTransformer)


class TestTransformerPipeline:
    """Tests for TransformerPipeline ordering via AutobotsAssemble (the intended API)."""

    def test_chained_inverse_reverses_order(self):
        """Chained log10 → normal_score via AutobotsAssemble inverse round-trips."""
        from pyemu.emulators.transformers import AutobotsAssemble

        np.random.seed(42)
        df = pd.DataFrame({"a": np.abs(np.random.normal(5, 2, 80)) + 1})

        ab = AutobotsAssemble(df.copy())
        ab.apply("log10", columns=["a"])
        ab.apply("normal_score", columns=["a"])

        inversed = ab.inverse()
        np.testing.assert_allclose(inversed["a"].values, df["a"].values, atol=0.5)

    def test_multi_column_pipeline(self):
        """Pipeline with different transforms on different columns."""
        from pyemu.emulators.transformers import AutobotsAssemble

        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.abs(np.random.normal(5, 2, 50)) + 1,
            "b": np.random.normal(0, 1, 50),
        })

        ab = AutobotsAssemble(df.copy())
        ab.apply("log10", columns=["a"])
        ab.apply("standard_scaler", columns=["b"])

        # "a" should be log-transformed
        np.testing.assert_allclose(ab.df["a"].values, np.log10(df["a"].values), atol=1e-10)
        # "b" should be standardized
        assert np.abs(ab.df["b"].mean()) < 0.1

        inversed = ab.inverse()
        np.testing.assert_allclose(inversed["a"].values, df["a"].values, atol=1e-5)
        np.testing.assert_allclose(inversed["b"].values, df["b"].values, atol=1e-5)


class TestAutobotsAssemble:
    """Tests for AutobotsAssemble high-level API."""

    def test_apply_and_inverse(self):
        """apply() builds pipeline; inverse() reverses it."""
        from pyemu.emulators.transformers import AutobotsAssemble

        np.random.seed(42)
        df = pd.DataFrame({"a": np.abs(np.random.normal(5, 2, 60)) + 1, "b": np.random.normal(0, 1, 60)})
        ab = AutobotsAssemble(df.copy())
        ab.apply("log10", columns=["a"])

        # Internal df should now be log-transformed
        np.testing.assert_allclose(ab.df["a"].values, np.log10(df["a"].values), atol=1e-10)
        # Column b unchanged
        np.testing.assert_allclose(ab.df["b"].values, df["b"].values, atol=1e-10)

        inversed = ab.inverse()
        np.testing.assert_allclose(inversed["a"].values, df["a"].values, atol=1e-5)

    def test_transform_external(self):
        """transform() applies fitted pipeline to new data."""
        from pyemu.emulators.transformers import AutobotsAssemble

        np.random.seed(42)
        train = pd.DataFrame({"a": np.random.normal(0, 1, 100)})
        ab = AutobotsAssemble(train.copy())
        ab.apply("normal_score", columns=["a"])

        test = pd.DataFrame({"a": np.random.normal(0, 1, 10)})
        transformed = ab.transform(test)
        assert transformed.shape == test.shape
        # Values should be different from input
        assert not np.allclose(transformed["a"].values, test["a"].values)

    def test_inverse_on_external_df(self):
        """inverse_on_external_df applies inverse to data not seen during fit."""
        from pyemu.emulators.transformers import AutobotsAssemble

        np.random.seed(42)
        df = pd.DataFrame({"a": np.abs(np.random.normal(5, 2, 60)) + 1})
        ab = AutobotsAssemble(df.copy())
        ab.apply("log10", columns=["a"])

        # Create external transformed data
        external = ab.transform(df)
        inversed = ab.inverse_on_external_df(external)
        np.testing.assert_allclose(inversed["a"].values, df["a"].values, atol=1e-5)


class TestRowWiseMinMaxScaler:
    """Tests for RowWiseMinMaxScaler (additional to existing test_row_wise_minmax_scaler)."""

    def test_fit_groups_subset(self):
        """fit_groups controls which columns determine row-wise min/max."""
        from pyemu.emulators.transformers import RowWiseMinMaxScaler

        df = pd.DataFrame({
            "history_1": [0.0, 5.0],
            "history_2": [10.0, 15.0],
            "forecast_1": [100.0, 200.0],  # much larger — should NOT affect scaling params
        })
        groups = {"ts": ["history_1", "history_2", "forecast_1"]}
        fit_groups = {"ts": ["history_1", "history_2"]}

        scaler = RowWiseMinMaxScaler(feature_range=(-1, 1), groups=groups, fit_groups=fit_groups)
        scaler.fit(df)

        # Row 0: min=0, max=10 from history columns
        row_min, row_max = scaler.row_params["ts"]
        assert row_min.iloc[0] == 0.0
        assert row_max.iloc[0] == 10.0

    def test_zero_variance_row(self):
        """Row where all values are identical does not cause division by zero."""
        from pyemu.emulators.transformers import RowWiseMinMaxScaler

        df = pd.DataFrame({
            "a": [5.0, 1.0],
            "b": [5.0, 2.0],  # Row 0: constant (5, 5)
        })
        groups = {"g": ["a", "b"]}
        scaler = RowWiseMinMaxScaler(feature_range=(-1, 1), groups=groups)
        transformed = scaler.fit_transform(df)
        assert np.all(np.isfinite(transformed.values))

        inversed = scaler.inverse_transform(transformed)
        # Row 1 (non-constant) should round-trip
        np.testing.assert_allclose(inversed.iloc[1].values, df.iloc[1].values, atol=1e-10)


class TestLog10Transformer:
    """Additional tests for Log10Transformer beyond existing test_log10_transformer."""

    def test_positive_values_no_shift(self):
        """Positive values produce no shift."""
        from pyemu.emulators.transformers import Log10Transformer

        df = pd.DataFrame({"a": [1.0, 10.0, 100.0]})
        t = Log10Transformer(columns=["a"])
        result = t.fit_transform(df)
        np.testing.assert_allclose(result["a"].values, [0.0, 1.0, 2.0], atol=1e-10)
        assert t.shifts["a"] == 0

    def test_negative_shift_roundtrip(self):
        """Columns with negatives get shifted and inverse round-trips."""
        from pyemu.emulators.transformers import Log10Transformer

        df = pd.DataFrame({"a": [-5.0, 0.0, 10.0]})
        t = Log10Transformer(columns=["a"])
        transformed = t.fit_transform(df)
        inversed = t.inverse_transform(transformed)
        np.testing.assert_allclose(inversed["a"].values, df["a"].values, atol=1e-5)

    def test_untouched_columns(self):
        """Columns not in 'columns' list remain unchanged."""
        from pyemu.emulators.transformers import Log10Transformer

        df = pd.DataFrame({"a": [1.0, 10.0], "b": [42.0, 99.0]})
        t = Log10Transformer(columns=["a"])
        result = t.fit_transform(df)
        np.testing.assert_array_equal(result["b"].values, df["b"].values)


# ===========================================================================
# PRIORITY 3 — Base class / file generation tests
# ===========================================================================


class TestBaseWriteTemplateFile:
    """Tests for Emulator._write_template_file."""

    def test_template_file_format(self, tmp_path):
        """Generated .tpl file has ptf header and correct parameter markers."""
        from pyemu.emulators.base import Emulator

        emu = Emulator(verbose=False)
        par_df = pd.DataFrame(
            {"parnme": ["p_0", "p_1"], "parval1": [0.0, 1.0]},
            index=["p_0", "p_1"],
        )
        path = str(tmp_path / "test.tpl")
        emu._write_template_file(par_df, path)

        with open(path) as f:
            lines = f.readlines()

        assert lines[0].strip() == "ptf ~"
        assert lines[1].strip() == "parnme,parval1"
        # Each parameter should have markers
        for i, pname in enumerate(["p_0", "p_1"]):
            assert f"~   {pname}   ~" in lines[i + 2]


class TestBaseWriteInstructionFile:
    """Tests for Emulator._write_instruction_file."""

    def test_instruction_file_format(self, tmp_path):
        """Generated .ins file has pif header and valid instruction grammar."""
        from pyemu.emulators.base import Emulator

        emu = Emulator(verbose=False)
        obs_df = pd.DataFrame(
            {"obsnme": ["obs0", "obs1"], "obsval": [1.0, 2.0], "weight": [1.0, 1.0], "obgnme": ["g", "g"]},
            index=["obs0", "obs1"],
        )
        path = str(tmp_path / "test.ins")
        emu._write_instruction_file(obs_df, path)

        with open(path) as f:
            lines = f.readlines()

        assert lines[0].strip() == "pif ~"
        assert lines[1].strip() == "l1"  # skip header
        for i, oname in enumerate(["obs0", "obs1"]):
            assert f"!{oname}!" in lines[i + 2]


class TestBaseWriteInputFile:
    """Tests for Emulator._write_input_file."""

    def test_input_file_readable(self, tmp_path):
        """Written input file is readable by pd.read_csv and values match."""
        from pyemu.emulators.base import Emulator

        emu = Emulator(verbose=False)
        par_df = pd.DataFrame(
            {"parnme": ["p_0", "p_1"], "parval1": [3.14, -2.71]},
            index=["p_0", "p_1"],
        )
        path = str(tmp_path / "input.csv")
        emu._write_input_file(par_df, path)

        result = pd.read_csv(path, index_col=0)
        np.testing.assert_allclose(result["parval1"].values, par_df["parval1"].values, atol=1e-10)


class TestBaseWriteOutputFile:
    """Tests for Emulator._write_output_file."""

    def test_output_file_format(self, tmp_path):
        """Written output file has header and correct values."""
        from pyemu.emulators.base import Emulator

        emu = Emulator(verbose=False)
        obs_df = pd.DataFrame(
            {"obsnme": ["obs0", "obs1"], "obsval": [1.5, -0.5], "weight": [1.0, 0.0], "obgnme": ["g", "g"]},
            index=["obs0", "obs1"],
        )
        path = str(tmp_path / "output.csv")
        emu._write_output_file(obs_df, path)

        result = pd.read_csv(path)
        assert list(result.columns) == ["obsnme", "simval"]
        assert list(result["obsnme"]) == ["obs0", "obs1"]
        np.testing.assert_allclose(result["simval"].values, [1.5, -0.5], atol=1e-10)


class TestBaseUpdateParameterData:
    """Tests for Emulator._update_parameter_data."""

    def test_merge_values(self):
        """Emulator par_df values are merged into pst par_df."""
        from pyemu.emulators.base import Emulator

        emu = Emulator(verbose=False)
        pst_par_df = pd.DataFrame(
            {"parnme": ["p_0", "p_1"], "parval1": [0.0, 0.0], "parlbnd": [0.0, 0.0],
             "parubnd": [0.0, 0.0], "pargp": ["x", "x"]},
            index=["p_0", "p_1"],
        )
        par_df = pd.DataFrame(
            {"parnme": ["p_0", "p_1"], "parval1": [1.0, 2.0], "parlbnd": [-10.0, -10.0],
             "parubnd": [10.0, 10.0], "pargp": ["dsi", "dsi"]},
            index=["p_0", "p_1"],
        )
        result = emu._update_parameter_data(pst_par_df, par_df)
        assert result.loc["p_0", "parval1"] == 1.0
        assert result.loc["p_1", "pargp"] == "dsi"
        assert result.loc["p_0", "parlbnd"] == -10.0


class TestBaseUpdateObservationData:
    """Tests for Emulator._update_observation_data."""

    def test_merge_values(self):
        """Emulator obs_df values are merged into pst obs_df."""
        from pyemu.emulators.base import Emulator

        emu = Emulator(verbose=False)
        pst_obs_df = pd.DataFrame(
            {"obsnme": ["o0", "o1"], "obsval": [0.0, 0.0], "weight": [0.0, 0.0], "obgnme": ["x", "x"]},
            index=["o0", "o1"],
        )
        obs_df = pd.DataFrame(
            {"obsnme": ["o0", "o1"], "obsval": [5.0, 6.0], "weight": [1.0, 2.0], "obgnme": ["grp", "grp"]},
            index=["o0", "o1"],
        )
        result = emu._update_observation_data(pst_obs_df, obs_df)
        assert result.loc["o0", "obsval"] == 5.0
        assert result.loc["o1", "weight"] == 2.0
        assert result.loc["o0", "obgnme"] == "grp"


class TestBaseValidateTransforms:
    """Tests for Emulator._validate_transforms error handling."""

    def test_not_list_raises(self):
        from pyemu.emulators.base import Emulator
        emu = Emulator(verbose=False)
        with pytest.raises(ValueError, match="list"):
            emu._validate_transforms("not a list")

    def test_not_dict_raises(self):
        from pyemu.emulators.base import Emulator
        emu = Emulator(verbose=False)
        with pytest.raises(ValueError, match="dict"):
            emu._validate_transforms(["not a dict"])

    def test_missing_type_raises(self):
        from pyemu.emulators.base import Emulator
        emu = Emulator(verbose=False)
        with pytest.raises(ValueError, match="type"):
            emu._validate_transforms([{"columns": ["a"]}])

    def test_columns_not_list_raises(self):
        from pyemu.emulators.base import Emulator
        emu = Emulator(verbose=False)
        with pytest.raises(ValueError, match="columns"):
            emu._validate_transforms([{"type": "log10", "columns": "not_a_list"}])

    def test_valid_transforms_pass(self):
        from pyemu.emulators.base import Emulator
        emu = Emulator(verbose=False)
        # Should not raise
        emu._validate_transforms([
            {"type": "log10", "columns": ["a", "b"]},
            {"type": "normal_score"},
        ])


# ===========================================================================
# GROUNDWORK FIX REGRESSION TESTS (DSIVC refactor preconditions)
# ===========================================================================


class TestBaseLowercaseIntake:
    """Fix base-lowercase: Emulator._lowercase_intake (base.py) lowercases all
    name-keyed state (data columns, transform 'columns' lists) at intake."""

    def test_base_lowercase_data_and_transform_columns(self):
        """Mixed-case data columns and transform 'columns' are lowercased."""
        import pyemu
        from pyemu.emulators.base import Emulator

        emu = Emulator(transforms=[{"type": "log10", "columns": ["FLOW", "HEAD"]}],
                       verbose=False)
        emu.data = pd.DataFrame({"FLOW": [1.0, 10.0], "HEAD": [100.0, 1000.0]})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pyemu.PyemuWarning)
            returned = emu._prepare_training_data()

        assert list(emu.data.columns) == ["flow", "head"]
        assert emu.transforms[0]["columns"] == ["flow", "head"]
        assert list(returned.columns) == ["flow", "head"]

    def test_base_lowercase_warns_once_with_mapping(self):
        """A single PyemuWarning naming the FLOW->flow mapping is emitted."""
        import pyemu
        from pyemu.emulators.base import Emulator

        emu = Emulator(verbose=False)
        emu.data = pd.DataFrame({"FLOW": [1.0, 2.0], "head": [3.0, 4.0]})

        with pytest.warns(pyemu.PyemuWarning,
                          match="lowercasing emulator data column names") as record:
            emu._prepare_training_data()

        msgs = [str(w.message) for w in record
                if "lowercasing emulator data column names" in str(w.message)]
        assert len(msgs) == 1
        assert "FLOW->flow" in msgs[0]

    def test_base_no_warn_when_already_lowercase(self):
        """No PyemuWarning when data/transform columns are already lowercase."""
        import pyemu
        from pyemu.emulators.base import Emulator

        emu = Emulator(transforms=[{"type": "log10", "columns": ["flow"]}],
                       verbose=False)
        emu.data = pd.DataFrame({"flow": [1.0, 10.0], "head": [2.0, 3.0]})

        with warnings.catch_warnings():
            warnings.simplefilter("error", pyemu.PyemuWarning)
            # must not raise (i.e. must not warn) for already-lowercase intake
            emu._prepare_training_data()

    def test_base_lowercase_collision_raises(self):
        """Columns that collide after lowercasing raise instead of silently
        producing duplicate column names."""
        from pyemu.emulators.base import Emulator

        emu = Emulator(verbose=False)
        emu.data = pd.DataFrame({"FLOW": [1.0, 2.0], "flow": [3.0, 4.0]})

        with pytest.raises(ValueError, match="duplicate"):
            emu._prepare_training_data()


class TestGPRLowercaseIntake:
    """Fix base-lowercase (GPR part): input/output names and prediction-input
    columns are lowercased so uppercase callers still align."""

    @staticmethod
    def _linear_df(cols=("x", "y")):
        x = np.linspace(0.0, 10.0, 20)
        y = 2.0 * x + 1.0
        return pd.DataFrame({cols[0]: x, cols[1]: y})

    def test_gpr_lowercases_input_output_names(self):
        """GPR lowercases input_names/output_names and data columns at init."""
        import pyemu
        from pyemu.emulators import GPR

        df = self._linear_df(cols=("IN_A", "OUT_X"))
        df["IN_B"] = np.linspace(-1.0, 1.0, 20)
        df = df[["IN_A", "IN_B", "OUT_X"]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pyemu.PyemuWarning)
            gpr = GPR(data=df, input_names=["IN_A", "IN_B"],
                      output_names=["OUT_X"], verbose=False)

        assert gpr.input_names == ["in_a", "in_b"]
        assert gpr.output_names == ["out_x"]
        assert list(gpr.data.columns) == ["in_a", "in_b", "out_x"]

    def test_gpr_predict_aligns_uppercase_input(self):
        """predict() lowercases uppercase X columns so they align with input_names."""
        from pyemu.emulators import GPR

        df = self._linear_df()  # lowercase columns x, y
        gpr = GPR(data=df, input_names=["x"], output_names=["y"],
                  n_restarts_optimizer=0, verbose=False)
        gpr.fit()

        X_upper = pd.DataFrame({"X": df["x"].values})  # uppercase input column
        pred = gpr.predict(X_upper)
        assert list(pred.columns) == ["y"]
        assert pred.shape == (df.shape[0], 1)

    def test_gpr_predict_lowercases_output_columns_for_runstore(self):
        """Built from UPPERCASE-named data, predict() returns lowercase columns."""
        import pyemu
        from pyemu.emulators import GPR

        df = self._linear_df(cols=("X", "Y"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pyemu.PyemuWarning)
            gpr = GPR(data=df, input_names=["X"], output_names=["Y"],
                      n_restarts_optimizer=0, verbose=False)
            gpr.fit()
            pred = gpr.predict(df[["X"]])

        assert all(c == c.lower() for c in pred.columns)
        assert list(pred.columns) == ["y"]


class TestDSILowercaseIntake:
    """Fix base-lowercase (DSI part): DSI overrides _prepare_training_data
    without calling super, so it must invoke _lowercase_intake itself; obs
    names in observation_data must follow the same lowercase namespace."""

    @staticmethod
    def _upper_data():
        np.random.seed(11)
        return pd.DataFrame(np.random.uniform(1.0, 10.0, size=(60, 4)),
                            columns=["OBS_A", "OBS_B", "OBS_C", "OBS_D"])

    def test_dsi_lowercases_data_and_transform_columns(self):
        """DSI built from UPPERCASE data lowercases data and transform columns."""
        import pyemu
        from pyemu.emulators import DSI

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pyemu.PyemuWarning)
            dsi = DSI(data=self._upper_data(),
                      transforms=[{"type": "log10", "columns": ["OBS_A"]}],
                      verbose=False)

        assert list(dsi.data.columns) == ["obs_a", "obs_b", "obs_c", "obs_d"]
        assert dsi.transforms[0]["columns"] == ["obs_a"]
        assert all(c == c.lower() for c in dsi.data_transformed.columns)

    def test_dsi_predict_output_names_lowercase(self):
        """predict() output names are lowercase (the runstore .rns alignment)."""
        import pyemu
        from pyemu.emulators import DSI

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pyemu.PyemuWarning)
            dsi = DSI(data=self._upper_data(), verbose=False)
            dsi.fit()
            pred = dsi.predict(pd.Series(np.zeros(dsi.pmat.shape[1])))

        assert list(pred.index) == ["obs_a", "obs_b", "obs_c", "obs_d"]

    def test_dsi_lowercases_observation_data_names(self):
        """observation_data obsnme/index are lowercased at intake so truth
        lookups stay aligned with the lowercased data columns."""
        import pyemu
        from pyemu.emulators import DSI

        data = self._upper_data()
        obsdata = pd.DataFrame(
            {"obsnme": data.columns, "obsval": data.mean().values,
             "weight": 1.0, "obgnme": "obgnme"},
            index=data.columns,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pyemu.PyemuWarning)
            dsi = DSI(data=data, pst=obsdata, verbose=False)

        assert list(dsi.observation_data.index) == ["obs_a", "obs_b", "obs_c", "obs_d"]
        assert list(dsi.observation_data.obsnme) == ["obs_a", "obs_b", "obs_c", "obs_d"]


class TestDSIPrepareWritesPst:
    """Fix dsi-pst-write: DSI.prepare_pestpp writes a complete dsi.pst to t_d."""

    @staticmethod
    def _data():
        np.random.seed(7)
        data = pd.DataFrame(np.random.normal(size=(80, 8)),
                            columns=[f"obs{i}" for i in range(8)])
        obsdata = pd.DataFrame(
            {"obsnme": data.columns, "obsval": data.mean().values,
             "weight": 1.0, "obgnme": "obgnme"},
            index=data.columns,
        )
        return data, obsdata

    def test_dsi_prepare_pestpp_writes_pst(self, tmp_path):
        """prepare_pestpp leaves a loadable dsi.pst with npar==nobs==n data-space pars."""
        import pyemu
        from pyemu.emulators import DSI

        data, obsdata = self._data()
        dsi = DSI(data=data, pst=obsdata, verbose=False)
        dsi.fit()

        td = tmp_path / "template_dsi"
        pst = dsi.prepare_pestpp(td, observation_data=obsdata, use_runstor=False)

        pst_path = os.path.join(td, "dsi.pst")
        assert os.path.exists(pst_path)
        loaded = pyemu.Pst(pst_path)
        n_pars = dsi.pmat.shape[1]
        assert loaded.npar == n_pars
        assert loaded.nobs == data.shape[1]

    def test_dsi_prepare_pestpp_runstor_writes_pst(self, tmp_path):
        """Under use_runstor=True the same write path produces a loadable dsi.pst."""
        import pyemu
        from pyemu.emulators import DSI

        data, obsdata = self._data()
        dsi = DSI(data=data, pst=obsdata, verbose=False)
        dsi.fit()

        td = tmp_path / "template_dsi_rs"
        dsi.prepare_pestpp(td, observation_data=obsdata, use_runstor=True)

        pst_path = os.path.join(td, "dsi.pst")
        assert os.path.exists(pst_path)
        loaded = pyemu.Pst(pst_path)
        assert loaded.pestpp_options["parcov"] == "dsi.unc"

    def test_dsi_caller_write_overrides_prepare_write(self, tmp_path):
        """A caller's later write cleanly overwrites the auto-written dsi.pst."""
        import pyemu
        from pyemu.emulators import DSI

        data, obsdata = self._data()
        dsi = DSI(data=data, pst=obsdata, verbose=False)
        dsi.fit()

        td = tmp_path / "template_dsi_over"
        returned = dsi.prepare_pestpp(td, observation_data=obsdata, use_runstor=True)
        returned.control_data.noptmax = 1
        returned.pestpp_options["ies_num_reals"] = 10
        returned.write(os.path.join(td, "dsi.pst"), version=2)

        loaded = pyemu.Pst(os.path.join(td, "dsi.pst"))
        assert loaded.control_data.noptmax == 1
        assert str(loaded.pestpp_options["ies_num_reals"]) == "10"


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
class TestDSIAEConfigurePstObservationData:
    """Fix dsiae-f1: DSIAE._configure_pst_object accepts and applies the
    observation_data kwarg the base hook caller always passes."""

    @staticmethod
    def _fit_dsiae(latent_dim=3):
        from pyemu.emulators import DSIAE
        np.random.seed(0)
        data = pd.DataFrame(np.random.normal(size=(60, 10)),
                            columns=[f"obs{i}" for i in range(10)])
        obsdata = pd.DataFrame(
            {"obsnme": data.columns, "obsval": data.mean().values,
             "weight": 1.0, "obgnme": "obgnme"},
            index=data.columns,
        )
        dsiae = DSIAE(data=data, latent_dim=latent_dim, verbose=False)
        dsiae.fit(epochs=2, batch_size=16, early_stopping=False)
        return dsiae, data, obsdata

    @staticmethod
    def _build_pst(dsiae):
        import pyemu
        par_df = dsiae._get_emulator_parameters()
        obs_df = dsiae._get_emulator_observations()
        pst = pyemu.Pst.from_par_obs_names(par_names=par_df.parnme.tolist(),
                                           obs_names=obs_df.obsnme.tolist())
        pst.parameter_data = dsiae._update_parameter_data(pst.parameter_data, par_df)
        pst.observation_data = dsiae._update_observation_data(pst.observation_data, obs_df)
        return pst

    def test_dsiae_configure_pst_accepts_observation_data(self, tmp_path):
        """The observation_data kwarg no longer raises TypeError; jcb artifacts written."""
        dsiae, _data, obsdata = self._fit_dsiae()
        pst = self._build_pst(dsiae)
        td = str(tmp_path / "dsiae_cfg")
        os.makedirs(td, exist_ok=True)

        dsiae._configure_pst_object(pst, None, observation_data=obsdata, t_d=td)

        assert pst.pestpp_options["ies_parameter_ensemble"] == "latent_prior.jcb"
        assert os.path.exists(os.path.join(td, "latent_prior.jcb"))

    def test_dsiae_configure_pst_applies_observation_data(self, tmp_path):
        """Supplied obsval/weight are actually applied via the base delegation."""
        dsiae, data, _obsdata = self._fit_dsiae()
        pst = self._build_pst(dsiae)
        td = str(tmp_path / "dsiae_cfg_apply")
        os.makedirs(td, exist_ok=True)

        # observation_data whose obsval/weight differ from the emulator defaults
        custom = pd.DataFrame(
            {"obsnme": data.columns,
             "obsval": np.arange(data.shape[1], dtype=float) + 0.5,
             "weight": np.arange(data.shape[1], dtype=float) + 1.0,
             "obgnme": "obgnme"},
            index=data.columns,
        )
        dsiae._configure_pst_object(pst, None, observation_data=custom, t_d=td)

        common = list(data.columns)
        applied = pst.observation_data.loc[common, ["obsval", "weight"]]
        np.testing.assert_allclose(applied["obsval"].values, custom["obsval"].values)
        np.testing.assert_allclose(applied["weight"].values, custom["weight"].values)


class TestLatentIndexOrdering:
    """Fix helpers-f13: dsi_runstore_forward_run orders run-store columns by the
    trailing integer of the parameter name (robust to dsi_par0000 names)."""

    @staticmethod
    def _trailing_int_key(name):
        """Replicates the fixed _latent_index sort key for the unit-level check."""
        import re
        m = re.search(r"(\d+)$", name)
        if m is None:
            raise ValueError(f"no trailing integer in {name}")
        return int(m.group(1))

    def test_latent_index_ordering_unit(self):
        """Trailing-integer sort yields increasing latent order for all name styles."""
        dsi_names = ["p_0", "p_1", "p_10", "p_11", "p_2", "p_3"]
        dsiae_names = ["dsi_par0000", "dsi_par0001", "dsi_par0010",
                       "dsi_par0011", "dsi_par0002", "dsi_par0003"]
        sv_names = ["sv_12", "sv_0", "sv_3", "sv_1"]

        for names, expected in [
            (dsi_names, [0, 1, 2, 3, 10, 11]),
            (dsiae_names, [0, 1, 2, 3, 10, 11]),
            (sv_names, [0, 1, 3, 12]),
        ]:
            ordered = sorted(names, key=self._trailing_int_key)
            assert [self._trailing_int_key(n) for n in ordered] == expected

        # a name with no trailing digit must raise, matching the fixed helper
        with pytest.raises(ValueError):
            self._trailing_int_key("dsi_par")

    def test_dsi_runstore_forward_run_embedded_import_re(self):
        """The generated forward_run.py body carries its own 'import re'.

        The standalone script header (Emulator._write_forward_run_script_body)
        imports only sys/os/pandas/numpy/traceback/pickle — NOT re — so the
        embedded dsi_runstore_forward_run source must import re itself.
        """
        import inspect
        from pyemu.utils.helpers import dsi_runstore_forward_run

        src = inspect.getsource(dsi_runstore_forward_run)
        assert "import re" in src

        # and the generated standalone header does not provide re for it
        from pyemu.emulators.base import Emulator
        import pyemu.utils.helpers as H
        emu = Emulator(verbose=False)
        out = str(Path.cwd() / "_unit_fr_check.py")
        try:
            emu._write_forward_run_script_body(
                out,
                [H.dsi_forward_run, H.dsi_file_forward_run, H.dsi_runstore_forward_run],
                "dsi_runstore_forward_run",
                "pst_name='dsi'",
            )
            text = Path(out).read_text()
        finally:
            if os.path.exists(out):
                os.remove(out)

        header = text.split("# Source for", 1)[0]
        assert "import re" not in header
        assert "import re" in text  # only the embedded function body supplies it


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
class TestDSIRunstoreForwardRunOrdering:
    """Fix helpers-f13 (integration): dsi_runstore_forward_run runs on
    DSIAE-style dsi_parNNNN names without ValueError and orders pvals correctly."""

    def test_dsi_runstore_forward_run_dsiae_names(self, tmp_path, monkeypatch):
        import pyemu.emulators as emu_mod
        import pyemu.utils.helpers as H
        from pyemu.emulators import DSIAE
        from pyemu.utils.helpers import dsi_runstore_forward_run

        latent_dim = 11
        np.random.seed(0)
        data = pd.DataFrame(np.random.normal(size=(60, 10)),
                            columns=[f"obs{i}" for i in range(10)])
        dsiae = DSIAE(data=data, latent_dim=latent_dim, verbose=False)
        dsiae.fit(epochs=2, batch_size=16, early_stopping=False)

        # file order is NOT latent order (dsi_par0010 sits before dsi_par0002)
        par_names_file = ["dsi_par0000", "dsi_par0001", "dsi_par0010", "dsi_par0002",
                          "dsi_par0003", "dsi_par0004", "dsi_par0005", "dsi_par0006",
                          "dsi_par0007", "dsi_par0008", "dsi_par0009"]
        obs_names = list(data.columns)
        n_runs = 3
        pvals_block = pd.DataFrame(np.random.normal(size=(n_runs, latent_dim)),
                                   columns=par_names_file)

        captured = {}

        class FakeRS:
            def __init__(self, fname):
                pass

            @staticmethod
            def file_info(fname):
                return {"n_runs": n_runs}, list(par_names_file), list(obs_names)

            def get_data(self):
                df = pd.DataFrame({"run_status": [0] * n_runs,
                                   "run_pos": [0] * n_runs})
                for c in par_names_file:
                    df[c] = pvals_block[c].values
                for c in obs_names:
                    df[c] = 0.0
                return df

            def update(self, df):
                captured["df"] = df.copy()

        # an empty dsi.rns so the real os.path.exists check passes
        monkeypatch.chdir(tmp_path)
        (tmp_path / "dsi.rns").write_bytes(b"")
        monkeypatch.setattr(H, "RunStor", FakeRS)
        monkeypatch.setattr(emu_mod, "DSIAE",
                            type("X", (), {"load": staticmethod(lambda p: dsiae)}))

        # must NOT raise ValueError on the dsi_parNNNN names
        dsi_runstore_forward_run(ws=str(tmp_path), pst_name="dsi")

        written = captured["df"]
        sorted_names = sorted(par_names_file,
                              key=TestLatentIndexOrdering._trailing_int_key)
        expected = dsiae.predict(pvals_block.loc[:, sorted_names])
        np.testing.assert_allclose(written.loc[:, obs_names].values,
                                   expected.loc[:, obs_names].values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
