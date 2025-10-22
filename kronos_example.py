#!/usr/bin/env python3
"""
Benchmark Kronos inference optimizations on synthetic multivariate time-series data.

The script mirrors the Toto benchmarking utility: it instantiates a lightweight
Kronos tokenizer/predictor pair, generates deterministic synthetic market data,
and times a set of inference strategies (baseline eager execution, inference_mode,
mixed precision autocast, FlashAttention-style SDP kernels, torch.compile, etc.).
Each strategy is compared against the FP32 baseline to quantify latency gains and
the numerical drift introduced by the optimization. Results can optionally be
appended to a Markdown log for long-term tracking.
"""
from __future__ import annotations

import argparse
import contextlib
import math
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from model.kronos import (  # type: ignore[import]
    Kronos,
    KronosTokenizer,
    auto_regressive_inference,
    calc_time_stamps,
)


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
    "ffn_dropout_p": 0.1,
    "attn_dropout_p": 0.0,
    "resid_dropout_p": 0.1,
    "token_dropout_p": 0.0,
    "learn_te": True,
}


@dataclass
class OptimizationStrategy:
    """Description of a Kronos inference optimization."""

    name: str
    description: str
    inference_mode: bool = False
    autocast_dtype: Optional[torch.dtype] = None
    compile: bool = False
    compile_mode: Optional[str] = None
    compile_fullgraph: bool = True
    enable_flash_sdp: Optional[bool] = None
    enable_math_sdp: Optional[bool] = None
    enable_mem_efficient_sdp: Optional[bool] = None
    allow_tf32: Optional[bool] = None
    notes: str = ""
    requires_cuda: bool = False
    skip_if_unsupported: bool = True

    def check_support(self, device: torch.device) -> Tuple[bool, str]:
        if self.requires_cuda and device.type != "cuda":
            return False, "requires CUDA device"
        if self.autocast_dtype in {torch.bfloat16, torch.float16} and device.type != "cuda":
            return False, f"{self.autocast_dtype} autocast requires CUDA"
        if self.compile and not hasattr(torch, "compile"):
            return False, "torch.compile unavailable in this PyTorch build"
        return True, ""


@dataclass
class OptimizationResult:
    name: str
    description: str
    latency_ms: Optional[float]
    speedup_vs_baseline: Optional[float]
    mean_abs_diff: Optional[float]
    max_abs_diff: Optional[float]
    relative_mae_pct: Optional[float]
    notes: str
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Kronos inference optimizations.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device for inference.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--batch-size", type=int, default=2, help="Synthetic batch size.")
    parser.add_argument("--context-length", type=int, default=64, help="Context window length.")
    parser.add_argument("--prediction-length", type=int, default=16, help="Prediction horizon.")
    parser.add_argument("--max-context", type=int, default=128, help="Maximum context window during autoregression.")
    parser.add_argument("--clip", type=float, default=5.0, help="Clip magnitude used before tokenization.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling parameter.")
    parser.add_argument("--top-p", type=float, default=0.99, help="Top-p sampling parameter.")
    parser.add_argument("--sample-count", type=int, default=1, help="Number of Monte Carlo trajectories.")
    parser.add_argument("--warmup-iters", type=int, default=1, help="Warmup iterations before measuring latency.")
    parser.add_argument("--output-md", type=str, default=None, help="Optional Markdown log file to append results.")
    parser.add_argument("--include-half", action="store_true", help="Include fp16 autocast strategy.")
    return parser.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cuda" or (arg == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class SyntheticBatch:
    series: torch.Tensor
    x_stamp: torch.Tensor
    y_stamp: torch.Tensor


def generate_synthetic_batch(
    *,
    batch_size: int,
    context_length: int,
    prediction_length: int,
    device: torch.device,
    clip: float,
) -> SyntheticBatch:
    """Create deterministic synthetic OHLCV-style data and timestamp features."""
    d_in = DEFAULT_TOKENIZER_CONFIG["d_in"]
    base_ts = pd.Timestamp("2025-01-02 09:30:00")
    total = context_length + prediction_length
    timeline = base_ts + pd.to_timedelta(np.arange(total), unit="min")

    price_trend = np.linspace(0.0, 1.0, context_length, dtype=np.float32)
    harmonic = np.sin(np.linspace(0, 4 * math.pi, context_length, dtype=np.float32))

    series_list = []
    for batch_idx in range(batch_size):
        rng = np.random.default_rng(seed=10 + batch_idx)
        base = price_trend + 0.1 * harmonic + 0.05 * rng.standard_normal(context_length)
        open_price = base + 0.01 * rng.standard_normal(context_length)
        close_price = base + 0.01 * rng.standard_normal(context_length)
        high_price = np.maximum(open_price, close_price) + np.abs(rng.standard_normal(context_length)) * 0.02
        low_price = np.minimum(open_price, close_price) - np.abs(rng.standard_normal(context_length)) * 0.02
        volume = np.abs(rng.standard_normal(context_length)) * 1000 + 100
        amount = volume * (open_price + close_price) / 2.0
        stacked = np.stack(
            [open_price, high_price, low_price, close_price, volume, amount],
            axis=-1,
        )
        series_list.append(stacked.astype(np.float32))

    series = torch.from_numpy(np.stack(series_list, axis=0))
    mean = series.mean(dim=1, keepdim=True)
    std = series.std(dim=1, keepdim=True).clamp_min(1e-5)
    series = torch.clamp((series - mean) / std, min=-clip, max=clip)

    time_df = calc_time_stamps(pd.Series(timeline))
    x_stamp = torch.from_numpy(time_df.iloc[:context_length].values).float()
    y_stamp = torch.from_numpy(time_df.iloc[context_length:].values).float()

    x_stamp = x_stamp.unsqueeze(0).repeat(batch_size, 1, 1)
    y_stamp = y_stamp.unsqueeze(0).repeat(batch_size, 1, 1)

    return SyntheticBatch(
        series=series.to(device),
        x_stamp=x_stamp.to(device),
        y_stamp=y_stamp.to(device),
    )


def instantiate_modules(
    *,
    tokenizer_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[KronosTokenizer, Kronos]:
    tokenizer = KronosTokenizer(**DEFAULT_TOKENIZER_CONFIG)
    model = Kronos(**DEFAULT_MODEL_CONFIG)
    tokenizer.load_state_dict(tokenizer_state, strict=True)
    model.load_state_dict(model_state, strict=True)
    tokenizer.eval()
    model.eval()
    tokenizer.to(device)
    model.to(device)
    return tokenizer, model


@contextlib.contextmanager
def cuda_backend_context(
    *,
    enable_flash_sdp: Optional[bool] = None,
    enable_math_sdp: Optional[bool] = None,
    enable_mem_efficient_sdp: Optional[bool] = None,
    allow_tf32: Optional[bool] = None,
) -> Iterator[None]:
    if not torch.cuda.is_available():
        yield
        return

    flash_state: Optional[bool] = None
    math_state: Optional[bool] = None
    mem_state: Optional[bool] = None
    tf32_state: Optional[bool] = None
    have_sdp_api = hasattr(torch.backends.cuda, "sdp_kernel")

    if enable_flash_sdp is not None or enable_math_sdp is not None or enable_mem_efficient_sdp is not None:
        if not have_sdp_api:
            raise RuntimeError("torch.backends.cuda.sdp_kernel is unavailable in this build.")
        flash_state = torch.backends.cuda.flash_sdp_enabled()
        math_state = torch.backends.cuda.math_sdp_enabled()
        mem_state = torch.backends.cuda.mem_efficient_sdp_enabled()
        torch.backends.cuda.sdp_kernel(
            enable_flash=flash_state if enable_flash_sdp is None else enable_flash_sdp,
            enable_math=math_state if enable_math_sdp is None else enable_math_sdp,
            enable_mem_efficient=mem_state if enable_mem_efficient_sdp is None else enable_mem_efficient_sdp,
        )

    if allow_tf32 is not None:
        tf32_state = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    try:
        yield
    finally:
        if tf32_state is not None:
            torch.backends.cuda.matmul.allow_tf32 = tf32_state
        if flash_state is not None or math_state is not None or mem_state is not None:
            torch.backends.cuda.sdp_kernel(
                enable_flash=flash_state if flash_state is not None else torch.backends.cuda.flash_sdp_enabled(),
                enable_math=math_state if math_state is not None else torch.backends.cuda.math_sdp_enabled(),
                enable_mem_efficient=(
                    mem_state if mem_state is not None else torch.backends.cuda.mem_efficient_sdp_enabled()
                ),
            )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def kronos_forecast(
    tokenizer: KronosTokenizer,
    model: Kronos,
    batch: SyntheticBatch,
    *,
    prediction_length: int,
    max_context: int,
    clip: float,
    temperature: float,
    top_k: int,
    top_p: float,
    sample_count: int,
) -> torch.Tensor:
    preds = auto_regressive_inference(
        tokenizer,
        model,
        batch.series,
        batch.x_stamp,
        batch.y_stamp,
        max_context=max_context,
        pred_len=prediction_length,
        clip=clip,
        T=temperature,
        top_k=top_k,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
    )
    preds_t = torch.from_numpy(preds).float()
    return preds_t[:, -prediction_length:, :]


def run_strategy(
    *,
    strategy: OptimizationStrategy,
    tokenizer_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
    batch: SyntheticBatch,
    device: torch.device,
    prediction_length: int,
    max_context: int,
    clip: float,
    temperature: float,
    top_k: int,
    top_p: float,
    sample_count: int,
    warmup_iters: int,
    seed: int,
) -> Tuple[str, Optional[torch.Tensor], Optional[float], str]:
    supported, reason = strategy.check_support(device)
    if not supported:
        return "skipped", None, None, reason

    seed_everything(seed)
    tokenizer, model = instantiate_modules(
        tokenizer_state=tokenizer_state,
        model_state=model_state,
        device=device,
    )

    compile_handles: List[torch.nn.Module] = []
    dynamo_module = None
    previous_capture_scalar_outputs = None

    previous_allow_unspec = None
    previous_recompile_limit = None

    try:
        if strategy.compile:
            compile_kwargs = {}
            if strategy.compile_mode is not None:
                compile_kwargs["mode"] = strategy.compile_mode
            compile_kwargs["fullgraph"] = strategy.compile_fullgraph
            compile_kwargs["dynamic"] = True

            try:
                import torch._dynamo as dynamo_module  # type: ignore[import-not-found]
            except Exception:
                dynamo_module = None

            if dynamo_module is not None:
                previous_capture_scalar_outputs = dynamo_module.config.capture_scalar_outputs
                dynamo_module.config.capture_scalar_outputs = True
                if hasattr(dynamo_module.config, "allow_unspec_int_on_nn_module"):
                    previous_allow_unspec = dynamo_module.config.allow_unspec_int_on_nn_module
                    dynamo_module.config.allow_unspec_int_on_nn_module = True
                if hasattr(dynamo_module.config, "recompile_limit"):
                    previous_recompile_limit = dynamo_module.config.recompile_limit
                    dynamo_module.config.recompile_limit = max(128, previous_recompile_limit)

            try:
                inductor_module = getattr(torch, "_inductor", None)
                previous_cudagraphs = None
                if inductor_module is not None:
                    inductor_config = inductor_module.config
                    if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraphs"):
                        previous_cudagraphs = inductor_config.triton.cudagraphs
                        inductor_config.triton.cudagraphs = False
                else:
                    inductor_module = None

                model.decode_s1 = torch.compile(model.decode_s1, **compile_kwargs)  # type: ignore[method-assign]
                model.decode_s2 = torch.compile(model.decode_s2, **compile_kwargs)  # type: ignore[method-assign]

                if inductor_module is not None and previous_cudagraphs is not None:
                    inductor_module.config.triton.cudagraphs = previous_cudagraphs
                if dynamo_module is not None:
                    if previous_allow_unspec is not None:
                        dynamo_module.config.allow_unspec_int_on_nn_module = previous_allow_unspec
                    if previous_recompile_limit is not None:
                        dynamo_module.config.recompile_limit = previous_recompile_limit
            except Exception as compile_err:  # noqa: BLE001
                raw_err = str(compile_err).strip()
                if not raw_err:
                    raw_err = compile_err.__class__.__name__
                if "CUDAGraphs" in raw_err:
                    raw_err = "CUDAGraph output reuse detected during torch.compile"
                if inductor_module is not None and previous_cudagraphs is not None:
                    inductor_module.config.triton.cudagraphs = previous_cudagraphs
                if dynamo_module is not None:
                    if previous_allow_unspec is not None:
                        dynamo_module.config.allow_unspec_int_on_nn_module = previous_allow_unspec
                    if previous_recompile_limit is not None:
                        dynamo_module.config.recompile_limit = previous_recompile_limit
                msg = f"torch.compile unavailable: {raw_err}"
                msg = " ".join(msg.splitlines())
                if len(msg) > 240:
                    msg = msg[:239] + "…"
                return "skipped", None, None, msg

        contexts: List[contextlib.AbstractContextManager] = []
        if strategy.inference_mode:
            contexts.append(torch.inference_mode())
        else:
            contexts.append(torch.no_grad())
        if strategy.autocast_dtype is not None:
            contexts.append(torch.autocast(device_type=device.type, dtype=strategy.autocast_dtype))
        contexts.append(
            cuda_backend_context(
                enable_flash_sdp=strategy.enable_flash_sdp,
                enable_math_sdp=strategy.enable_math_sdp,
                enable_mem_efficient_sdp=strategy.enable_mem_efficient_sdp,
                allow_tf32=strategy.allow_tf32,
            )
        )

        try:
            with contextlib.ExitStack() as stack:
                for ctx in contexts:
                    stack.enter_context(ctx)
                for _ in range(max(warmup_iters, 0)):
                    seed_everything(seed)
                    _ = kronos_forecast(
                        tokenizer,
                        model,
                        batch,
                        prediction_length=prediction_length,
                        max_context=max_context,
                        clip=clip,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        sample_count=sample_count,
                    )
                synchronize(device)
                seed_everything(seed)
                start = time.perf_counter()
                preds = kronos_forecast(
                    tokenizer,
                    model,
                    batch,
                    prediction_length=prediction_length,
                    max_context=max_context,
                    clip=clip,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    sample_count=sample_count,
                )
                synchronize(device)
                latency_ms = (time.perf_counter() - start) * 1000.0
        except Exception as err:  # noqa: BLE001
            raw_err = str(err).strip()
            if not raw_err:
                raw_err = err.__class__.__name__
            if "CUDAGraphs" in raw_err:
                raw_err = "CUDAGraph output reuse detected during torch.compile runtime"
            msg = f"{strategy.name} failed: {raw_err}"
            msg = " ".join(msg.splitlines())
            if len(msg) > 240:
                msg = msg[:239] + "…"
            return "skipped", None, None, msg

        return "ok", preds, latency_ms, ""
    finally:
        for handle in compile_handles:
            del handle
        if dynamo_module is not None and previous_capture_scalar_outputs is not None:
            dynamo_module.config.capture_scalar_outputs = previous_capture_scalar_outputs
        if dynamo_module is not None:
            if previous_allow_unspec is not None:
                dynamo_module.config.allow_unspec_int_on_nn_module = previous_allow_unspec
            if previous_recompile_limit is not None:
                dynamo_module.config.recompile_limit = previous_recompile_limit


def compute_quality_metrics(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
) -> Tuple[float, float, float]:
    diff = (candidate - baseline).abs()
    mean_abs_diff = diff.mean().item()
    max_abs_diff = diff.max().item()
    baseline_abs_mean = baseline.abs().mean().item()
    relative_mae_pct = 0.0 if math.isclose(baseline_abs_mean, 0.0) else (mean_abs_diff / baseline_abs_mean) * 100.0
    return mean_abs_diff, max_abs_diff, relative_mae_pct


def build_strategies(args: argparse.Namespace, device: torch.device) -> List[OptimizationStrategy]:
    strategies = [
        OptimizationStrategy(
            name="baseline_fp32",
            description="Standard eager inference in float32.",
            notes="Reference configuration.",
        ),
        OptimizationStrategy(
            name="inference_mode",
            description="torch.inference_mode() to disable autograd overhead.",
            inference_mode=True,
        ),
        OptimizationStrategy(
            name="flash_sdp",
            description="Prioritize FlashAttention kernels where supported.",
            inference_mode=True,
            enable_flash_sdp=True,
            enable_math_sdp=False,
            enable_mem_efficient_sdp=False,
            allow_tf32=True,
            requires_cuda=True,
            notes="Forces FlashAttention-style kernels via torch.backends.cuda.sdp_kernel.",
        ),
    ]

    if device.type == "cuda":
        strategies.append(
            OptimizationStrategy(
                name="bf16_autocast",
                description="CUDA autocast to bfloat16.",
                inference_mode=True,
                autocast_dtype=torch.bfloat16,
                notes="BF16 autocast for Ampere+ GPUs.",
                requires_cuda=True,
            )
        )
        strategies.append(
            OptimizationStrategy(
                name="fp32_compile_max_autotune_flash",
                description="FP32 inference with torch.compile(max-autotune) and FlashAttention kernels.",
                inference_mode=True,
                compile=True,
                compile_mode="max-autotune",
                enable_flash_sdp=True,
                enable_math_sdp=False,
                enable_mem_efficient_sdp=False,
                allow_tf32=True,
                notes="Compiled FP32 path; retains full precision while enabling FlashAttention.",
                requires_cuda=True,
            )
        )
        strategies.append(
            OptimizationStrategy(
                name="bf16_compile_reduce_overhead",
                description="bf16 autocast + torch.compile(mode='reduce-overhead').",
                inference_mode=True,
                autocast_dtype=torch.bfloat16,
                compile=True,
                compile_mode="reduce-overhead",
                notes="Compiles decode_s1/decode_s2 graph; excludes compile time from latency.",
                requires_cuda=True,
            )
        )
        strategies.append(
            OptimizationStrategy(
                name="bf16_compile_max_autotune",
                description="bf16 autocast + torch.compile(mode='max-autotune').",
                inference_mode=True,
                autocast_dtype=torch.bfloat16,
                compile=True,
                compile_mode="max-autotune",
                notes="Aggressive autotuning for kernels.",
                requires_cuda=True,
            )
        )
        if args.include_half:
            strategies.append(
                OptimizationStrategy(
                    name="fp16_autocast",
                    description="CUDA autocast to float16.",
                    inference_mode=True,
                    autocast_dtype=torch.float16,
                    notes="Potentially higher speed but larger numerical drift.",
                    requires_cuda=True,
                )
            )

    return strategies


def format_markdown_table(results: Sequence[OptimizationResult]) -> str:
    headers = [
        "Strategy",
        "Latency (ms)",
        "Speedup",
        "Mean |Δ|",
        "Max |Δ|",
        "Rel MAE %",
        "Status",
        "Notes",
    ]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for result in results:
        row = [
            result.name,
            f"{result.latency_ms:.3f}" if result.latency_ms is not None else "n/a",
            f"{result.speedup_vs_baseline:.2f}x" if result.speedup_vs_baseline is not None else "n/a",
            f"{result.mean_abs_diff:.3e}" if result.mean_abs_diff is not None else "n/a",
            f"{result.max_abs_diff:.3e}" if result.max_abs_diff is not None else "n/a",
            f"{result.relative_mae_pct:.2f}%" if result.relative_mae_pct is not None else "n/a",
            result.status,
            result.notes,
        ]
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[INFO] Using device: {device}")

    seed_everything(args.seed)

    batch = generate_synthetic_batch(
        batch_size=args.batch_size,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        device=device,
        clip=args.clip,
    )

    # Persist initial weights to reuse across strategies
    base_tokenizer = KronosTokenizer(**DEFAULT_TOKENIZER_CONFIG)
    base_model = Kronos(**DEFAULT_MODEL_CONFIG)
    base_tokenizer.eval()
    base_model.eval()
    tokenizer_state = {k: v.detach().cpu() for k, v in base_tokenizer.state_dict().items()}
    model_state = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}

    strategies = build_strategies(args, device)
    results: List[OptimizationResult] = []
    baseline_prediction: Optional[torch.Tensor] = None
    baseline_latency_ms: Optional[float] = None

    for idx, strategy in enumerate(strategies):
        print(f"[INFO] Running strategy {idx + 1}/{len(strategies)}: {strategy.name}")
        status, prediction, latency_ms, message = run_strategy(
            strategy=strategy,
            tokenizer_state=tokenizer_state,
            model_state=model_state,
            batch=batch,
            device=device,
            prediction_length=args.prediction_length,
            max_context=args.max_context,
            clip=args.clip,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_count=args.sample_count,
            warmup_iters=args.warmup_iters,
            seed=args.seed,
        )

        if status == "skipped":
            results.append(
                OptimizationResult(
                    name=strategy.name,
                    description=strategy.description,
                    latency_ms=None,
                    speedup_vs_baseline=None,
                    mean_abs_diff=None,
                    max_abs_diff=None,
                    relative_mae_pct=None,
                    notes=message or strategy.notes,
                    status="skipped",
                )
            )
            print(f"[WARN] Skipped {strategy.name}: {message}")
            continue

        assert prediction is not None and latency_ms is not None
        prediction = prediction.to(dtype=torch.float32)

        if baseline_prediction is None:
            baseline_prediction = prediction
            baseline_latency_ms = latency_ms
            mean_abs_diff = 0.0
            max_abs_diff = 0.0
            relative_mae_pct = 0.0
            speedup = 1.0
        else:
            mean_abs_diff, max_abs_diff, relative_mae_pct = compute_quality_metrics(baseline_prediction, prediction)
            speedup = (baseline_latency_ms / latency_ms) if (baseline_latency_ms and latency_ms > 0) else None

        result = OptimizationResult(
            name=strategy.name,
            description=strategy.description,
            latency_ms=latency_ms,
            speedup_vs_baseline=speedup,
            mean_abs_diff=mean_abs_diff,
            max_abs_diff=max_abs_diff,
            relative_mae_pct=relative_mae_pct,
            notes=strategy.notes,
            status="ok",
        )
        results.append(result)
        print(
            "[INFO] "
            f"{strategy.name}: latency={latency_ms:.3f}ms speedup="
            f"{(speedup if speedup is not None else float('nan')):.2f}x "
            f"mean_abs_diff={mean_abs_diff:.3e} max_abs_diff={max_abs_diff:.3e} "
            f"rel_mae={relative_mae_pct:.2f}%"
        )

    print("\n=== Kronos Inference Optimization Summary ===")
    md_table = format_markdown_table(results)
    print(md_table)

    if args.output_md:
        header = (
            f"\n\n### Benchmark run (device={device}, torch={torch.__version__}, seed={args.seed}, "
            f"batch={args.batch_size}, context={args.context_length}, pred={args.prediction_length})\n\n"
        )
        with open(args.output_md, "a", encoding="utf-8") as handle:
            handle.write(header)
            handle.write(md_table)
            handle.write("\n")
        print(f"[INFO] Appended results to {args.output_md}")


if __name__ == "__main__":
    main()
