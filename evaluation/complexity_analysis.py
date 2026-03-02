"""
evaluation/complexity_analysis.py

Computational Complexity and Scalability Analysis

Measures and reports:
    1. Training time for each component
    2. Inference time per sample (latency)
    3. Memory footprint
    4. Parameter count comparison (vs BERT, DistilBERT)
    5. Scalability: inference time vs input size

Q1 journals require quantitative evidence of the "lightweight" claim.
"""

import time
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tracemalloc

RESULTS_DIR = "results"


def measure_training_time(X_train, y_train, detection_agent,
                           verification_agent=None, real_statements=None,
                           speaker_cred=None):
    """Measure training/fitting time for each component."""
    print("\n[Complexity] Measuring training times...")
    results = {}

    # Detection Agent training time
    tracemalloc.start()
    t0 = time.perf_counter()
    detection_agent.train(X_train, y_train)
    t1 = time.perf_counter()
    current, peak_det = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results["detection_agent_train"] = {
        "time_seconds":   round(t1 - t0, 3),
        "peak_memory_MB": round(peak_det / 1024 / 1024, 2),
    }
    print(f"  Detection Agent training:  {t1-t0:.3f}s  |  Peak RAM: {peak_det/1024/1024:.2f} MB")

    # Verification Agent fitting time
    if verification_agent and real_statements:
        tracemalloc.start()
        t0 = time.perf_counter()
        verification_agent.fit(real_statements, speaker_cred)
        t1 = time.perf_counter()
        current, peak_ver = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results["verification_agent_fit"] = {
            "time_seconds":   round(t1 - t0, 3),
            "peak_memory_MB": round(peak_ver / 1024 / 1024, 2),
        }
        print(f"  Verification Agent fitting:{t1-t0:.3f}s  |  Peak RAM: {peak_ver/1024/1024:.2f} MB")

    return results


def measure_inference_time(X_test, detection_agent, verification_agent,
                           decision_agent, speakers=None, n_repeats=3):
    """Measure per-sample inference latency."""
    print("\n[Complexity] Measuring inference latency...")
    if speakers is None:
        speakers = ["unknown"] * len(X_test)

    results = {}

    # Detection Agent inference
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        d_results = detection_agent.get_detection_scores(X_test)
        times.append(time.perf_counter() - t0)
    det_total = np.mean(times)
    det_per_sample = det_total / len(X_test) * 1000  # ms

    results["detection_inference"] = {
        "total_seconds":   round(det_total, 4),
        "per_sample_ms":   round(det_per_sample, 4),
        "n_samples":       len(X_test),
    }
    print(f"  Detection Agent:    {det_total:.3f}s total  |  {det_per_sample:.4f} ms/sample")

    # Verification Agent inference
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        v_results = verification_agent.verify_batch(X_test, speakers)
        times.append(time.perf_counter() - t0)
    ver_total = np.mean(times)
    ver_per_sample = ver_total / len(X_test) * 1000

    results["verification_inference"] = {
        "total_seconds":   round(ver_total, 4),
        "per_sample_ms":   round(ver_per_sample, 4),
        "n_samples":       len(X_test),
    }
    print(f"  Verification Agent: {ver_total:.3f}s total  |  {ver_per_sample:.4f} ms/sample")

    # Decision Agent inference
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = decision_agent.decide_batch(d_results, v_results)
        times.append(time.perf_counter() - t0)
    dec_total = np.mean(times)
    dec_per_sample = dec_total / len(X_test) * 1000

    results["decision_inference"] = {
        "total_seconds":   round(dec_total, 4),
        "per_sample_ms":   round(dec_per_sample, 4),
        "n_samples":       len(X_test),
    }
    print(f"  Decision Agent:     {dec_total:.3f}s total  |  {dec_per_sample:.4f} ms/sample")

    # Full pipeline
    pipeline_per_sample = det_per_sample + ver_per_sample + dec_per_sample
    results["full_pipeline"] = {
        "per_sample_ms":   round(pipeline_per_sample, 4),
        "throughput_per_sec": round(1000 / pipeline_per_sample, 1),
    }
    print(f"  Full Pipeline:      {pipeline_per_sample:.4f} ms/sample  "
          f"| Throughput: {1000/pipeline_per_sample:.0f} samples/sec")

    return results, d_results, v_results


def parameter_comparison():
    """Compare parameter counts with other models."""
    print("\n[Complexity] Parameter count comparison...")

    # L-MAS: TF-IDF matrix (10000 features * 2 classes for LR = 20000 params)
    # + LR bias = 2 params → ~20,002 total
    vocab_size = 10000
    lr_params  = vocab_size * 2 + 2  # weights + bias, binary

    comparison = {
        "L-MAS (This Work)": {
            "parameters":          lr_params,
            "parameters_millions": round(lr_params / 1e6, 4),
            "model_size_MB":       round(lr_params * 4 / 1024 / 1024, 3),
            "gpu_required":        False,
            "inference_type":      "CPU",
        },
        "BERT-base": {
            "parameters":          110_000_000,
            "parameters_millions": 110.0,
            "model_size_MB":       440.0,
            "gpu_required":        True,
            "inference_type":      "GPU (recommended)",
        },
        "DistilBERT": {
            "parameters":          66_000_000,
            "parameters_millions": 66.0,
            "model_size_MB":       260.0,
            "gpu_required":        True,
            "inference_type":      "GPU (recommended)",
        },
        "RoBERTa-base": {
            "parameters":          125_000_000,
            "parameters_millions": 125.0,
            "model_size_MB":       500.0,
            "gpu_required":        True,
            "inference_type":      "GPU required",
        },
    }

    print("\n  Model Parameter Comparison:")
    for model, info in comparison.items():
        print(f"  {model:25s}  {info['parameters_millions']:8.2f}M params  "
              f"{info['model_size_MB']:7.1f} MB  GPU={info['gpu_required']}")

    return comparison


def scalability_test(detection_agent, X_base, batch_sizes=(50,100,200,500,1000)):
    """Test inference time vs batch size."""
    print("\n[Complexity] Scalability test (batch size vs latency)...")
    results = {}

    for bsz in batch_sizes:
        if bsz > len(X_base):
            X_batch = X_base * (bsz // len(X_base) + 1)
        else:
            X_batch = X_base
        X_batch = X_batch[:bsz]

        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            detection_agent.get_detection_scores(X_batch)
            times.append(time.perf_counter() - t0)

        avg_time = np.mean(times)
        results[bsz] = {
            "total_seconds":  round(avg_time, 4),
            "per_sample_ms":  round(avg_time / bsz * 1000, 4),
        }
        print(f"  Batch size={bsz:5d}  Total={avg_time:.4f}s  Per-sample={avg_time/bsz*1000:.4f}ms")

    # Plot
    bsizes = list(results.keys())
    p_sample = [results[b]["per_sample_ms"] for b in bsizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bsizes, p_sample, "o-", color="#2563EB", lw=2)
    ax.set_xlabel("Batch Size (number of statements)")
    ax.set_ylabel("Per-sample Inference Time (ms)")
    ax.set_title("Scalability: Inference Latency vs. Batch Size", fontweight="bold")
    ax.fill_between(bsizes, p_sample, alpha=0.15, color="#2563EB")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "scalability_test.png"), dpi=150)
    plt.close()

    return results


def _plot_model_comparison(param_data):
    names  = list(param_data.keys())
    params = [param_data[n]["parameters_millions"] for n in names]
    colors = ["#DC2626" if "L-MAS" in n else "#93C5FD" for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, params, color=colors, alpha=0.88, edgecolor="white")
    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{p:.2f}M", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Parameters (Millions)")
    ax.set_title("Model Size Comparison: L-MAS vs. Transformer Models",
                 fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.001, 500)

    # Annotate L-MAS
    ax.text(0, params[0] * 3, "No GPU\nneeded", ha="center",
            fontsize=8.5, color="#DC2626", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "parameter_comparison.png"), dpi=150)
    plt.close()


def run_full_complexity_analysis(X_train, y_train, X_test,
                                 detection_agent, verification_agent,
                                 decision_agent, real_statements,
                                 speaker_cred, speakers_test):
    """Run all complexity analyses."""
    print("\n" + "="*60)
    print("  COMPLEXITY AND SCALABILITY ANALYSIS")
    print("="*60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    combined = {}

    combined["training_times"]  = measure_training_time(
        X_train, y_train, detection_agent,
        verification_agent, real_statements, speaker_cred
    )

    infer_results, d_res, v_res = measure_inference_time(
        X_test, detection_agent, verification_agent,
        decision_agent, speakers_test
    )
    combined["inference_times"] = infer_results

    combined["parameter_comparison"] = parameter_comparison()
    _plot_model_comparison(combined["parameter_comparison"])

    combined["scalability"] = scalability_test(detection_agent, X_test)

    with open(os.path.join(RESULTS_DIR, "complexity_analysis.json"), "w") as f:
        json.dump(combined, f, indent=4)

    print(f"\n[Complexity] All results saved → results/complexity_analysis.json")
    print(f"[Complexity] Plots: parameter_comparison.png, scalability_test.png")

    return combined, d_res, v_res
