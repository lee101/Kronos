#!/usr/bin/env python3
"""
Run a lightweight Kronos inference smoke test on a deterministic synthetic series.

The script instantiates a tiny Kronos tokenizer/model pair, feeds a simple arithmetic
progression (2, 4, 6, 8, ...) through the predictor, and prints both the generated
forecasts and the average latency across multiple runs. It mirrors the unit test
regression setup so that manual executions can double-check numerical stability and
performance after code changes.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.kronos import Kronos, KronosPredictor, KronosTokenizer  # type: ignore[import]

DEFAULT_TOKENIZER_CONFIG = {
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

DEFAULT_MODEL_CONFIG = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kronos inference smoke test.")
    parser.add_argument("--device", default="cpu", help="Target device (e.g. cpu, cuda:0).")
    parser.add_argument("--context-len", type=int, default=4, help="Number of historical steps.")
    parser.add_argument("--pred-len", type=int, default=2, help="Prediction horizon.")
    parser.add_argument("--clip", type=float, default=5.0, help="Clipping value before tokenization.")
    parser.add_argument("--max-context", type=int, default=32, help="Maximum context the model can attend to.")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed inference runs.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility.")
    parser.add_argument("--suppress-nvml-warning", action="store_true", help="Disable PyTorch NVML warning.")
    return parser.parse_args()


def build_components(device: str, clip: float, max_context: int, seed: int) -> KronosPredictor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tokenizer = KronosTokenizer(**DEFAULT_TOKENIZER_CONFIG)
    model = Kronos(**DEFAULT_MODEL_CONFIG)
    predictor = KronosPredictor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_context=max_context,
        clip=clip,
    )
    return predictor


def build_series(context_len: int, pred_len: int) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    total_len = context_len + pred_len
    full_series = np.arange(2.0, 2.0 * (total_len + 1), 2.0, dtype=np.float32)
    context = full_series[:context_len]

    frame = pd.DataFrame(
        {
            "open": context,
            "high": context + 0.2,
            "low": context - 0.2,
            "close": context + 0.1,
            "volume": np.full_like(context, 1000.0),
            "amount": (context + 0.1) * 1000.0,
        }
    )

    timeline = pd.date_range("2025-01-02 09:30", periods=total_len, freq="15min")
    x_timestamp = pd.Series(timeline[:context_len])
    y_timestamp = pd.Series(timeline[context_len:])
    return frame, x_timestamp, y_timestamp


def run_inference(
    predictor: KronosPredictor,
    frame: pd.DataFrame,
    x_timestamp: pd.Series,
    y_timestamp: pd.Series,
    pred_len: int,
    seed: int,
) -> pd.DataFrame:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return predictor.predict(
            df=frame,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_k=0,
            top_p=0.99,
            sample_count=1,
            verbose=False,
        )


def main() -> None:
    args = parse_args()

    if args.suppress_nvml_warning:
        os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK_DISABLE_WARNING", "1")

    predictor = build_components(args.device, args.clip, args.max_context, args.seed)
    frame, x_timestamp, y_timestamp = build_series(args.context_len, args.pred_len)

    warmup = run_inference(predictor, frame, x_timestamp, y_timestamp, args.pred_len, args.seed)

    timings: list[float] = []
    for _ in range(args.runs):
        start = perf_counter()
        run_inference(predictor, frame, x_timestamp, y_timestamp, args.pred_len, args.seed)
        timings.append((perf_counter() - start) * 1000.0)

    forecast = warmup.loc[:, ["open", "high", "low", "close", "volume", "amount"]]
    print("\nForecast for the arithmetic progression (columns: open/high/low/close/volume/amount):")
    print(forecast)

    mean_ms = float(np.mean(timings)) if timings else float("nan")
    p95_ms = float(np.percentile(timings, 95)) if len(timings) >= 1 else float("nan")

    print("\nLatency (ms) over {} runs:".format(len(timings)))
    print(f"  mean: {mean_ms:.3f} ms")
    print(f"  p95 : {p95_ms:.3f} ms")


if __name__ == "__main__":
    main()
