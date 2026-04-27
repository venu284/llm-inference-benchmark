#!/usr/bin/env python3
"""
Plot Charts - Visualization generation for benchmark results.
CSCI 8970 - LLM Inference Benchmarking
Authors: Venu Dattathreya Vemuru, Varshith Peddineni

Generates publication-quality figures:
  1. Latency comparison bar charts (per batch size)
  2. Throughput comparison bar charts
  3. Speedup heatmap (framework × config)
  4. Memory usage comparison
  5. TTFT vs sequence length
  6. Compile overhead bar chart
  7. Latency distribution boxplots
  8. Throughput scaling with batch size
  9. GPU utilization & power comparison
  10. Radar chart (multi-metric framework comparison)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.family'] = 'sans-serif'

# Color palette for frameworks
FW_COLORS = {
    'eager': '#4C72B0',
    'compile_default': '#55A868',
    'compile_reduce_overhead': '#C44E52',
    'compile_max_autotune': '#8172B2',
    'tensorrt': '#CCB974',
}

FW_LABELS = {
    'eager': 'PyTorch Eager',
    'compile_default': 'compile (default)',
    'compile_reduce_overhead': 'compile (reduce-OH)',
    'compile_max_autotune': 'compile (max-autotune)',
    'tensorrt': 'TensorRT',
}

FW_ORDER = ['eager', 'compile_default', 'compile_reduce_overhead',
            'compile_max_autotune', 'tensorrt']


def get_color(fw):
    return FW_COLORS.get(fw, '#999999')

def get_label(fw):
    return FW_LABELS.get(fw, fw)

def fw_sort_key(fw):
    return FW_ORDER.index(fw) if fw in FW_ORDER else 99


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['fw_label'] = df['framework'].map(get_label)
    return df


# =====================================================================
# CHART 1: Latency Comparison (Grouped Bar per Batch Size)
# =====================================================================
def plot_latency_by_batch(df, figures_dir):
    for bs in sorted(df['batch_size'].unique()):
        sub = df[df['batch_size'] == bs].sort_values(
            by=['seq_length', 'framework'],
            key=lambda x: x.map(lambda v: fw_sort_key(v) if v in FW_ORDER else v)
        )

        fig, ax = plt.subplots(figsize=(10, 5.5))
        seq_lengths = sorted(sub['seq_length'].unique())
        n_fw = len(sub['framework'].unique())
        width = 0.15
        x = np.arange(len(seq_lengths))

        for i, fw in enumerate(FW_ORDER):
            fw_data = sub[sub['framework'] == fw]
            if fw_data.empty:
                continue
            vals = [fw_data[fw_data['seq_length'] == sl]['latency_mean_ms'].values[0]
                    for sl in seq_lengths]
            bars = ax.bar(x + i * width, vals, width, label=get_label(fw),
                          color=get_color(fw), edgecolor='white', linewidth=0.5)
            # Add value labels on bars
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Sequence Length (tokens)')
        ax.set_ylabel('Mean Latency (ms)')
        ax.set_title(f'Inference Latency — Batch Size = {bs}', fontweight='bold')
        ax.set_xticks(x + width * (n_fw - 1) / 2)
        ax.set_xticklabels([str(s) for s in seq_lengths])
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        path = os.path.join(figures_dir, f'latency_batch{bs}.png')
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")


# =====================================================================
# CHART 2: Throughput Comparison (Grouped Bar per Batch Size)
# =====================================================================
def plot_throughput_by_batch(df, figures_dir):
    for bs in sorted(df['batch_size'].unique()):
        sub = df[df['batch_size'] == bs]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        seq_lengths = sorted(sub['seq_length'].unique())
        width = 0.15
        x = np.arange(len(seq_lengths))

        for i, fw in enumerate(FW_ORDER):
            fw_data = sub[sub['framework'] == fw]
            if fw_data.empty:
                continue
            vals = [fw_data[fw_data['seq_length'] == sl]['throughput_tok_per_sec'].values[0]
                    for sl in seq_lengths]
            bars = ax.bar(x + i * width, vals, width, label=get_label(fw),
                          color=get_color(fw), edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Sequence Length (tokens)')
        ax.set_ylabel('Throughput (tokens/sec)')
        ax.set_title(f'Inference Throughput — Batch Size = {bs}', fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([str(s) for s in seq_lengths])
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        path = os.path.join(figures_dir, f'throughput_batch{bs}.png')
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")


# =====================================================================
# CHART 3: Speedup Heatmap
# =====================================================================
def plot_speedup_heatmap(df, figures_dir):
    eager = df[df['framework'] == 'eager'].set_index(['batch_size', 'seq_length'])

    rows = []
    for _, row in df.iterrows():
        key = (row['batch_size'], row['seq_length'])
        if key in eager.index:
            sp = eager.loc[key, 'latency_mean_ms'] / row['latency_mean_ms']
            config_label = f"bs={row['batch_size']}\nseq={row['seq_length']}"
            rows.append({
                'framework': get_label(row['framework']),
                'config': config_label,
                'speedup': sp,
            })

    sp_df = pd.DataFrame(rows)
    pivot = sp_df.pivot(index='framework', columns='config', values='speedup')

    # Reorder
    fw_label_order = [get_label(fw) for fw in FW_ORDER if get_label(fw) in pivot.index]
    pivot = pivot.reindex(fw_label_order)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0,
                linewidths=0.5, ax=ax, vmin=0.8, vmax=1.6,
                cbar_kws={'label': 'Speedup vs Eager'})
    ax.set_title('Speedup vs PyTorch Eager (higher = faster)', fontweight='bold')
    ax.set_ylabel('')
    ax.set_xlabel('')

    path = os.path.join(figures_dir, 'speedup_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# =====================================================================
# CHART 4: Peak Memory Comparison
# =====================================================================
def plot_memory(df, figures_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, bs in enumerate(sorted(df['batch_size'].unique())):
        ax = axes[idx]
        sub = df[df['batch_size'] == bs]
        seq_lengths = sorted(sub['seq_length'].unique())
        width = 0.15
        x = np.arange(len(seq_lengths))

        for i, fw in enumerate(FW_ORDER):
            fw_data = sub[sub['framework'] == fw]
            if fw_data.empty:
                continue
            vals = [fw_data[fw_data['seq_length'] == sl]['peak_memory_mb'].values[0]
                    for sl in seq_lengths]
            ax.bar(x + i * width, vals, width, label=get_label(fw),
                   color=get_color(fw), edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Sequence Length')
        ax.set_title(f'Batch Size = {bs}', fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([str(s) for s in seq_lengths])
        ax.grid(axis='y', alpha=0.3)

        # 24GB VRAM line
        ax.axhline(y=24000, color='red', linestyle='--', alpha=0.5, label='24GB VRAM limit')

        if idx == 0:
            ax.set_ylabel('Peak Memory (MB)')

    axes[0].legend(fontsize=7, loc='upper left')
    fig.suptitle('GPU Memory Usage by Configuration', fontweight='bold', fontsize=13)
    fig.tight_layout()

    path = os.path.join(figures_dir, 'memory_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# =====================================================================
# CHART 5: TTFT vs Sequence Length
# =====================================================================
def plot_ttft(df, figures_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for fw in FW_ORDER:
        fw_data = df[(df['framework'] == fw) & (df['batch_size'] == 1)]
        if fw_data.empty:
            continue
        fw_data = fw_data.sort_values('seq_length')
        ax.plot(fw_data['seq_length'], fw_data['ttft_ms'],
                marker='o', linewidth=2, markersize=8,
                label=get_label(fw), color=get_color(fw))

    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Time to First Token (ms)')
    ax.set_title('TTFT vs Sequence Length (Batch Size = 1)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(sorted(df['seq_length'].unique()))

    path = os.path.join(figures_dir, 'ttft_vs_seqlen.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# =====================================================================
# CHART 6: Compilation Overhead
# =====================================================================
def plot_compile_overhead(df, figures_dir):
    compile_data = df.groupby('framework').agg({
        'compile_time_sec': 'first',
        'num_graph_breaks': 'first',
    }).reset_index()
    compile_data = compile_data[compile_data['compile_time_sec'] > 0]

    if compile_data.empty:
        return

    compile_data['fw_label'] = compile_data['framework'].map(get_label)
    compile_data = compile_data.sort_values(
        'framework', key=lambda x: x.map(fw_sort_key))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Compile time
    colors = [get_color(fw) for fw in compile_data['framework']]
    bars = ax1.barh(compile_data['fw_label'], compile_data['compile_time_sec'],
                    color=colors, edgecolor='white')
    for bar, val in zip(bars, compile_data['compile_time_sec']):
        ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}s', va='center', fontsize=10)
    ax1.set_xlabel('Compilation Time (seconds)')
    ax1.set_title('One-Time Compile Cost', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Graph breaks
    bars2 = ax2.barh(compile_data['fw_label'], compile_data['num_graph_breaks'],
                     color=colors, edgecolor='white')
    for bar, val in zip(bars2, compile_data['num_graph_breaks']):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'{int(val)}', va='center', fontsize=10)
    ax2.set_xlabel('Number of Graph Breaks')
    ax2.set_title('Graph Breaks (torch.compile)', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    fig.suptitle('Compilation Overhead Analysis', fontweight='bold', fontsize=13)
    fig.tight_layout()

    path = os.path.join(figures_dir, 'compile_overhead.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# =====================================================================
# CHART 7: Throughput Scaling with Batch Size
# =====================================================================
def plot_throughput_scaling(df, figures_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for fw in FW_ORDER:
        fw_data = df[(df['framework'] == fw) & (df['seq_length'] == 512)]
        if fw_data.empty:
            continue
        fw_data = fw_data.sort_values('batch_size')
        ax.plot(fw_data['batch_size'], fw_data['throughput_tok_per_sec'],
                marker='s', linewidth=2, markersize=8,
                label=get_label(fw), color=get_color(fw))

    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (tokens/sec)')
    ax.set_title('Throughput Scaling with Batch Size (seq=512)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(sorted(df['batch_size'].unique()))

    path = os.path.join(figures_dir, 'throughput_scaling.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# =====================================================================
# CHART 8: GPU Utilization & Power
# =====================================================================
def plot_gpu_power(df, figures_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    avg = df.groupby('framework').agg({
        'gpu_util_mean_pct': 'mean',
        'power_draw_mean_w': 'mean',
    }).reindex([fw for fw in FW_ORDER if fw in df['framework'].unique()])

    labels = [get_label(fw) for fw in avg.index]
    colors = [get_color(fw) for fw in avg.index]

    # GPU utilization
    bars1 = ax1.barh(labels, avg['gpu_util_mean_pct'], color=colors, edgecolor='white')
    for bar, val in zip(bars1, avg['gpu_util_mean_pct']):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center', fontsize=10)
    ax1.set_xlabel('GPU Utilization (%)')
    ax1.set_title('Average GPU Utilization', fontweight='bold')
    ax1.set_xlim(0, 105)
    ax1.grid(axis='x', alpha=0.3)

    # Power draw
    bars2 = ax2.barh(labels, avg['power_draw_mean_w'], color=colors, edgecolor='white')
    for bar, val in zip(bars2, avg['power_draw_mean_w']):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f'{val:.0f}W', va='center', fontsize=10)
    ax2.set_xlabel('Power Draw (Watts)')
    ax2.set_title('Average Power Consumption', fontweight='bold')
    ax2.axvline(x=350, color='red', linestyle='--', alpha=0.5, label='RTX 3090 TDP')
    ax2.legend(fontsize=8)
    ax2.grid(axis='x', alpha=0.3)

    fig.suptitle('GPU Resource Utilization', fontweight='bold', fontsize=13)
    fig.tight_layout()

    path = os.path.join(figures_dir, 'gpu_utilization_power.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# =====================================================================
# CHART 9: Per-Token Latency Heatmap
# =====================================================================
def plot_per_token_heatmap(df, figures_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, bs in enumerate(sorted(df['batch_size'].unique())):
        ax = axes[idx]
        sub = df[df['batch_size'] == bs]

        pivot = sub.pivot_table(
            index='framework', columns='seq_length',
            values='per_token_latency_ms'
        )
        fw_label_order = [fw for fw in FW_ORDER if fw in pivot.index]
        pivot = pivot.reindex(fw_label_order)
        pivot.index = [get_label(fw) for fw in pivot.index]

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                    linewidths=0.5, ax=ax,
                    cbar_kws={'label': 'ms/token'} if idx == 2 else {'label': ''})
        ax.set_title(f'Batch Size = {bs}', fontweight='bold')
        ax.set_ylabel('' if idx > 0 else 'Framework')
        ax.set_xlabel('Sequence Length')

    fig.suptitle('Per-Token Latency (ms/token)', fontweight='bold', fontsize=13)
    fig.tight_layout()

    path = os.path.join(figures_dir, 'per_token_latency_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# =====================================================================
# CHART 10: Summary Radar Chart
# =====================================================================
def plot_radar(df, figures_dir):
    # Aggregate per framework (batch=1, seq=512 as representative config)
    sub = df[(df['batch_size'] == 1) & (df['seq_length'] == 512)]
    if sub.empty:
        return

    categories = ['Throughput', 'Low Latency', 'Low Memory', 'Low TTFT', 'GPU Efficiency']
    N = len(categories)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Normalize: higher is better for all
    max_throughput = sub['throughput_tok_per_sec'].max()
    max_latency = sub['latency_mean_ms'].max()
    max_memory = sub['peak_memory_mb'].max()
    max_ttft = sub['ttft_ms'].max()
    max_util = sub['gpu_util_mean_pct'].max()

    for fw in FW_ORDER:
        row = sub[sub['framework'] == fw]
        if row.empty:
            continue
        row = row.iloc[0]

        values = [
            row['throughput_tok_per_sec'] / max_throughput,          # higher = better
            1 - (row['latency_mean_ms'] / max_latency),             # lower latency = better
            1 - (row['peak_memory_mb'] / max_memory),               # lower memory = better
            1 - (row['ttft_ms'] / max_ttft),                        # lower TTFT = better
            row['gpu_util_mean_pct'] / max_util,                    # higher util = better
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=get_label(fw),
                color=get_color(fw), markersize=6)
        ax.fill(angles, values, alpha=0.08, color=get_color(fw))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title('Framework Comparison (batch=1, seq=512)\nHigher = Better',
                 fontweight='bold', pad=20, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    path = os.path.join(figures_dir, 'radar_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# =====================================================================
# CHART 11: Latency P99 vs Mean scatter
# =====================================================================
def plot_p99_vs_mean(df, figures_dir):
    fig, ax = plt.subplots(figsize=(9, 6))

    for fw in FW_ORDER:
        fw_data = df[df['framework'] == fw]
        if fw_data.empty:
            continue
        ax.scatter(fw_data['latency_mean_ms'], fw_data['latency_p99_ms'],
                   label=get_label(fw), color=get_color(fw), s=80, alpha=0.8,
                   edgecolors='white', linewidth=0.5)

    # Diagonal line (P99 == mean)
    lims = [0, df['latency_p99_ms'].max() * 1.1]
    ax.plot(lims, lims, '--', color='gray', alpha=0.4, label='P99 = Mean')

    ax.set_xlabel('Mean Latency (ms)')
    ax.set_ylabel('P99 Latency (ms)')
    ax.set_title('Latency Consistency: P99 vs Mean', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    path = os.path.join(figures_dir, 'p99_vs_mean.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    csv_path = os.path.join(ROOT, 'results', 'benchmark_results.csv')
    figures_dir = os.path.join(ROOT, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"[ERROR] Results not found at {csv_path}")
        print("Run generate_synthetic_results.py or run_benchmark.py first.")
        sys.exit(1)

    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows. Generating charts...\n")

    plot_latency_by_batch(df, figures_dir)
    plot_throughput_by_batch(df, figures_dir)
    plot_speedup_heatmap(df, figures_dir)
    plot_memory(df, figures_dir)
    plot_ttft(df, figures_dir)
    plot_compile_overhead(df, figures_dir)
    plot_throughput_scaling(df, figures_dir)
    plot_gpu_power(df, figures_dir)
    plot_per_token_heatmap(df, figures_dir)
    plot_radar(df, figures_dir)
    plot_p99_vs_mean(df, figures_dir)

    print(f"\nAll charts saved to {figures_dir}/")
    print(f"Total: {len(os.listdir(figures_dir))} figures")


if __name__ == "__main__":
    main()
