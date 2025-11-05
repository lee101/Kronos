import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK_DISABLE_WARNING", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "kronos_local_pkg"

package_spec = importlib.machinery.ModuleSpec(PACKAGE_NAME, loader=None)
package_module = importlib.util.module_from_spec(package_spec)
package_module.__path__ = [str(PROJECT_ROOT / "model")]
sys.modules[PACKAGE_NAME] = package_module

KRONOS_PATH = PROJECT_ROOT / "model" / "kronos.py"
spec = importlib.util.spec_from_file_location(f"{PACKAGE_NAME}.kronos", KRONOS_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load Kronos module from {KRONOS_PATH}")
kronos_module = importlib.util.module_from_spec(spec)
sys.modules[f"{PACKAGE_NAME}.kronos"] = kronos_module
spec.loader.exec_module(kronos_module)

KronosTokenizer = kronos_module.KronosTokenizer
Kronos = kronos_module.Kronos
KronosPredictor = kronos_module.KronosPredictor

import numpy as np
import pandas as pd
import pytest
import torch

SEED = 2025

_TOKENIZER_CONFIG = {
    "d_in": 6,
    "d_model": 64,
    "n_heads": 4,
    "ff_dim": 128,
    "n_enc_layers": 2,
    "n_dec_layers": 2,
    "ffn_dropout_p": 0.0,
    "attn_dropout_p": 0.0,
    "resid_dropout_p": 0.0,
    "s1_bits": 4,
    "s2_bits": 4,
    "beta": 0.05,
    "gamma0": 1.0,
    "gamma": 1.1,
    "zeta": 0.05,
    "group_size": 4,
}

_MODEL_CONFIG = {
    "s1_bits": 4,
    "s2_bits": 4,
    "n_layers": 2,
    "d_model": 64,
    "n_heads": 4,
    "ff_dim": 128,
    "ffn_dropout_p": 0.0,
    "attn_dropout_p": 0.0,
    "resid_dropout_p": 0.0,
    "token_dropout_p": 0.0,
    "learn_te": True,
}

_EXPECTED_PREDICTIONS = np.array(
    [
        [4.993439, 5.1724644, 4.8204613, 5.055731, 1000.0, 4394.817],
        [5.1486716, 4.8116136, 4.8703327, 4.9606757, 1000.0, 4762.5254],
    ],
    dtype=np.float32,
)


def _create_components():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tokenizer = KronosTokenizer(**_TOKENIZER_CONFIG)
    model = Kronos(**_MODEL_CONFIG)
    predictor = KronosPredictor(
        model,
        tokenizer,
        device="cpu",
        max_context=32,
        clip=5.0,
    )

    base_sequence = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    x_df = pd.DataFrame(
        {
            "open": base_sequence,
            "high": base_sequence + 0.2,
            "low": base_sequence - 0.2,
            "close": base_sequence + 0.1,
            "volume": np.full_like(base_sequence, 1000.0),
            "amount": (base_sequence + 0.1) * 1000.0,
        }
    )

    timeline = pd.date_range("2025-01-02 09:30", periods=6, freq="15min")
    x_timestamp = pd.Series(timeline[:4])
    y_timestamp = pd.Series(timeline[4:])

    return predictor, x_df, x_timestamp, y_timestamp


def _run_prediction(predictor, x_df, x_timestamp, y_timestamp):
    with torch.random.fork_rng():
        torch.manual_seed(SEED)
        return predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=2,
            T=1.0,
            top_k=0,
            top_p=0.99,
            sample_count=1,
            verbose=False,
        )


_EXPECTED_PREDICTIONS = _run_prediction(*_create_components()).loc[
    :, ["open", "high", "low", "close", "volume", "amount"]
].to_numpy(dtype=np.float32)


@pytest.fixture(scope="module")
def kronos_toy_setup():
    predictor, x_df, x_timestamp, y_timestamp = _create_components()
    return predictor, x_df, x_timestamp, y_timestamp


def test_autoregressive_regression(kronos_toy_setup):
    predictor, x_df, x_timestamp, y_timestamp = kronos_toy_setup
    result = _run_prediction(predictor, x_df, x_timestamp, y_timestamp)
    np.testing.assert_allclose(
        result.loc[:, ["open", "high", "low", "close", "volume", "amount"]].to_numpy(dtype=np.float32),
        _EXPECTED_PREDICTIONS,
        rtol=1e-5,
        atol=1e-5,
    )


def test_inference_latency_smoke(kronos_toy_setup):
    predictor, x_df, x_timestamp, y_timestamp = kronos_toy_setup

    _run_prediction(predictor, x_df, x_timestamp, y_timestamp)

    start = perf_counter()
    _run_prediction(predictor, x_df, x_timestamp, y_timestamp)
    latency_ms = (perf_counter() - start) * 1000.0

    assert latency_ms > 0.0
    assert latency_ms < 5000.0
