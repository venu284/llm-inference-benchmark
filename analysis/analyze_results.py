#!/usr/bin/env python3
"""
Analyze Results - Aggregate analysis of benchmark data.
CSCI 8970 - LLM Inference Benchmarking
Authors: Venu Dattathreya Vemuru, Varshith Peddineni

Reads benchmark_results.csv and produces:
  - Framework comparison summary
  - Speedup tables relative to eager baseline
  - Memory overhead analysis
  - Compile cost analysis
  - Best config recommendations
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Frameworks: {df['framework'].unique().tolist()}")
    print(f"Batch sizes: {sorted(df['batch_size'].unique().tolist())}")
    print(f"Seq lengths: {sorted(df['seq_length'].unique().tolist())}")
    return df


def framework_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Average metrics per framework across all configs."""
    summary = df.groupby('framework').agg({
        'latency_mean_ms': 'mean',
        'latency_p99_ms': 'mean',
        'throughput_tok_per_sec': 'mean',
        'ttft_ms': 'mean',
        'peak_memory_mb': 'mean',
        'gpu_util_mean_pct': 'mean',
        'power_draw_mean_w': 'mean',
        'compile_time_sec': 'first',
        'num_graph_breaks': 'first',
    }).round(2)

    # Order frameworks logically
    fw_order = ['eager', 'compile_default', 'compile_reduce_overhead',
                'compile_max_autotune', 'tensorrt']
    existing = [fw for fw in fw_order if fw in summary.index]
    summary = summary.reindex(existing)

    return summary


def speedup_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute speedup of each framework vs eager for each (batch, seq) config."""
    eager = df[df['framework'] == 'eager'].set_index(['batch_size', 'seq_length'])

    rows = []
    for _, row in df.iterrows():
        key = (row['batch_size'], row['seq_length'])
        if key in eager.index:
            eager_lat = eager.loc[key, 'latency_mean_ms']
            speedup = eager_lat / row['latency_mean_ms']
            rows.append({
                'framework': row['framework'],
                'batch_size': row['batch_size'],
                'seq_length': row['seq_length'],
                'latency_ms': round(row['latency_mean_ms'], 1),
                'eager_latency_ms': round(eager_lat, 1),
                'speedup_vs_eager': round(speedup, 3),
                'throughput_tok_per_sec': round(row['throughput_tok_per_sec'], 1),
            })

    speedup_df = pd.DataFrame(rows)
    return speedup_df


def memory_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Memory overhead vs eager baseline."""
    eager_mem = df[df['framework'] == 'eager'].groupby(
        ['batch_size', 'seq_length'])['peak_memory_mb'].first()

    rows = []
    for _, row in df.iterrows():
        key = (row['batch_size'], row['seq_length'])
        if key in eager_mem.index:
            overhead = row['peak_memory_mb'] - eager_mem[key]
            pct = (overhead / eager_mem[key]) * 100
            rows.append({
                'framework': row['framework'],
                'batch_size': row['batch_size'],
                'seq_length': row['seq_length'],
                'peak_memory_mb': round(row['peak_memory_mb'], 0),
                'memory_overhead_mb': round(overhead, 0),
                'memory_overhead_pct': round(pct, 1),
            })

    return pd.DataFrame(rows)


def compile_cost_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze compilation overhead: time spent compiling vs time saved."""
    eager_lat = df[df['framework'] == 'eager'].set_index(
        ['batch_size', 'seq_length'])['latency_mean_ms']

    rows = []
    for fw in df['framework'].unique():
        if fw == 'eager':
            continue
        fw_data = df[df['framework'] == fw]
        compile_time = fw_data['compile_time_sec'].iloc[0]

        for _, row in fw_data.iterrows():
            key = (row['batch_size'], row['seq_length'])
            if key in eager_lat.index:
                time_saved_per_run_ms = eager_lat[key] - row['latency_mean_ms']
                time_saved_per_run_s = time_saved_per_run_ms / 1000.0
                if time_saved_per_run_s > 0:
                    breakeven_runs = compile_time / time_saved_per_run_s
                else:
                    breakeven_runs = float('inf')

                rows.append({
                    'framework': fw,
                    'batch_size': row['batch_size'],
                    'seq_length': row['seq_length'],
                    'compile_time_sec': compile_time,
                    'time_saved_per_run_sec': round(time_saved_per_run_s, 3),
                    'breakeven_runs': round(breakeven_runs, 0),
                })

    return pd.DataFrame(rows)


def best_configs(df: pd.DataFrame) -> dict:
    """Find best framework for each optimization goal."""
    results = {}

    # Best latency per (batch, seq) config
    idx = df.groupby(['batch_size', 'seq_length'])['latency_mean_ms'].idxmin()
    results['lowest_latency'] = df.loc[idx][
        ['framework', 'batch_size', 'seq_length', 'latency_mean_ms']
    ].to_dict('records')

    # Best throughput per config
    idx = df.groupby(['batch_size', 'seq_length'])['throughput_tok_per_sec'].idxmax()
    results['highest_throughput'] = df.loc[idx][
        ['framework', 'batch_size', 'seq_length', 'throughput_tok_per_sec']
    ].to_dict('records')

    # Lowest memory per config
    idx = df.groupby(['batch_size', 'seq_length'])['peak_memory_mb'].idxmin()
    results['lowest_memory'] = df.loc[idx][
        ['framework', 'batch_size', 'seq_length', 'peak_memory_mb']
    ].to_dict('records')

    return results


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'benchmark_results.csv'
    )

    if not os.path.exists(csv_path):
        print(f"[ERROR] Results not found at {csv_path}")
        print("Run generate_synthetic_results.py or run_benchmark.py first.")
        sys.exit(1)

    df = load_results(csv_path)

    # --- 1. Framework Summary ---
    print_section("1. Framework Summary (Averaged Across All Configs)")
    summary = framework_summary(df)
    print(summary.to_string())

    # --- 2. Speedup Table ---
    print_section("2. Speedup vs Eager Baseline")
    sp = speedup_table(df)
    pivot = sp.pivot_table(
        index='framework', columns=['batch_size', 'seq_length'],
        values='speedup_vs_eager'
    ).round(3)
    fw_order = ['eager', 'compile_default', 'compile_reduce_overhead',
                'compile_max_autotune', 'tensorrt']
    existing = [fw for fw in fw_order if fw in pivot.index]
    pivot = pivot.reindex(existing)
    print(pivot.to_string())

    avg_speedup = sp.groupby('framework')['speedup_vs_eager'].mean().round(3)
    print(f"\nAverage speedup across all configs:")
    for fw in existing:
        if fw in avg_speedup.index:
            print(f"  {fw:<30} {avg_speedup[fw]:.3f}x")

    # --- 3. Memory Analysis ---
    print_section("3. Memory Overhead vs Eager")
    mem = memory_analysis(df)
    mem_avg = mem.groupby('framework')['memory_overhead_mb'].mean().round(0)
    mem_pct = mem.groupby('framework')['memory_overhead_pct'].mean().round(1)
    for fw in existing:
        if fw in mem_avg.index:
            print(f"  {fw:<30} +{mem_avg[fw]:>6.0f} MB ({mem_pct[fw]:>+5.1f}%)")

    # --- 4. Compile Cost ---
    print_section("4. Compilation Cost Analysis")
    cc = compile_cost_analysis(df)
    if not cc.empty:
        cc_summary = cc.groupby('framework').agg({
            'compile_time_sec': 'first',
            'time_saved_per_run_sec': 'mean',
            'breakeven_runs': 'mean',
        }).round(1)
        for fw in existing:
            if fw in cc_summary.index and fw != 'eager':
                r = cc_summary.loc[fw]
                print(f"  {fw:<30} compile={r['compile_time_sec']:>6.1f}s  "
                      f"saves={r['time_saved_per_run_sec']:>5.3f}s/run  "
                      f"breakeven={r['breakeven_runs']:>6.0f} runs")

    # --- 5. Best Configs ---
    print_section("5. Best Framework Per Config")
    best = best_configs(df)
    print("\n  Lowest latency winner per config:")
    for rec in best['lowest_latency']:
        print(f"    batch={rec['batch_size']}, seq={rec['seq_length']}: "
              f"{rec['framework']} ({rec['latency_mean_ms']:.1f}ms)")

    print("\n  Highest throughput winner per config:")
    for rec in best['highest_throughput']:
        print(f"    batch={rec['batch_size']}, seq={rec['seq_length']}: "
              f"{rec['framework']} ({rec['throughput_tok_per_sec']:.1f} tok/s)")

    # --- 6. Key Findings ---
    print_section("6. Key Findings")

    # Average speedups
    eager_mean = df[df['framework'] == 'eager']['latency_mean_ms'].mean()
    for fw in ['compile_default', 'compile_reduce_overhead', 'compile_max_autotune']:
        fw_data = df[df['framework'] == fw]
        if fw_data.empty:
            continue
        fw_mean = fw_data['latency_mean_ms'].mean()
        sp_val = eager_mean / fw_mean
        print(f"  • {fw} achieves {sp_val:.2f}x average speedup over eager")

    # Memory cost of best compiler
    eager_mem = df[df['framework'] == 'eager']['peak_memory_mb'].mean()
    best_fw = 'compile_max_autotune'
    best_fw_data = df[df['framework'] == best_fw]
    if not best_fw_data.empty:
        best_mem = best_fw_data['peak_memory_mb'].mean()
        print(f"  • {best_fw} uses {best_mem - eager_mem:.0f} MB more memory than eager on average")

    # Best batch-size for throughput
    bs_thr = df.groupby('batch_size')['throughput_tok_per_sec'].mean()
    best_bs = bs_thr.idxmax()
    print(f"  • Batch size {best_bs} achieves highest average throughput ({bs_thr[best_bs]:.1f} tok/s)")

    # Seq length impact on TTFT
    ttft_by_seq = df[df['framework'] == 'eager'].groupby('seq_length')['ttft_ms'].mean()
    print(f"  • TTFT increases {ttft_by_seq.iloc[-1]/ttft_by_seq.iloc[0]:.1f}x "
          f"from seq={sorted(df['seq_length'].unique())[0]} to "
          f"seq={sorted(df['seq_length'].unique())[-1]} (eager)")

    print(f"\n{'='*70}")
    print(f"  Analysis complete.")
    print(f"{'='*70}")

    # Save analysis summary to CSV
    summary_path = os.path.join(
        os.path.dirname(csv_path), 'analysis_summary.csv'
    )
    summary.to_csv(summary_path)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
