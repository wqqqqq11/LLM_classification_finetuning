"""
Dataset analysis script for LLM classification competition.
Generates statistical reports and visualizations to inform modeling decisions.
"""

import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd


def load_data(train_path, test_path):
    """Load training and test datasets."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def analyze_basic_info(train_df, test_df, output_dir):
    """Generate basic dataset statistics."""
    report = {
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_columns": list(train_df.columns),
        "test_columns": list(test_df.columns),
        "missing_values_train": train_df.isnull().sum().to_dict(),
        "missing_values_test": test_df.isnull().sum().to_dict(),
    }

    output_file = output_dir / "basic_info.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def analyze_labels(train_df, output_dir):
    """Analyze label distribution in training set."""
    labels = ["winner_model_a", "winner_model_b", "winner_tie"]
    label_counts = {}
    label_ratios = {}

    for label in labels:
        count = train_df[label].sum()
        ratio = count / len(train_df)
        label_counts[label] = int(count)
        label_ratios[label] = round(ratio, 4)

    report = {
        "label_counts": label_counts,
        "label_ratios": label_ratios,
    }

    output_file = output_dir / "label_distribution.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def analyze_text_lengths(train_df, test_df, output_dir):
    """Analyze text length statistics."""
    text_cols = ["prompt", "response_a", "response_b"]

    def get_length_stats(df, col):
        lengths = df[col].fillna("").astype(str).str.len()
        return {
            "mean": round(lengths.mean(), 2),
            "median": round(lengths.median(), 2),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "std": round(lengths.std(), 2),
            "p95": round(lengths.quantile(0.95), 2),
            "p99": round(lengths.quantile(0.99), 2),
        }

    train_stats = {col: get_length_stats(train_df, col) for col in text_cols}
    test_stats = {col: get_length_stats(test_df, col) for col in text_cols}

    report = {
        "train": train_stats,
        "test": test_stats,
    }

    output_file = output_dir / "text_lengths.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def analyze_models(train_df, output_dir):
    """Analyze model identifier distribution."""
    model_a_counts = Counter(train_df["model_a"].dropna())
    model_b_counts = Counter(train_df["model_b"].dropna())

    all_models = set(model_a_counts.keys()) | set(model_b_counts.keys())
    model_presence = {}
    for model in all_models:
        model_presence[model] = {
            "as_model_a": model_a_counts.get(model, 0),
            "as_model_b": model_b_counts.get(model, 0),
        }

    report = {
        "unique_models": len(all_models),
        "model_list": sorted(list(all_models)),
        "model_presence": model_presence,
    }

    output_file = output_dir / "model_distribution.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def analyze_tie_samples(train_df, output_dir):
    """Detailed analysis of tie samples."""
    tie_df = train_df[train_df["winner_tie"] == 1]
    non_tie_df = train_df[train_df["winner_tie"] == 0]

    def compare_lengths(col):
        tie_lengths = tie_df[col].fillna("").astype(str).str.len()
        non_tie_lengths = non_tie_df[col].fillna("").astype(str).str.len()
        return {
            "tie_mean": round(tie_lengths.mean(), 2),
            "non_tie_mean": round(non_tie_lengths.mean(), 2),
        }

    report = {
        "tie_sample_count": len(tie_df),
        "tie_ratio": round(len(tie_df) / len(train_df), 4),
        "text_length_comparison": {
            "prompt": compare_lengths("prompt"),
            "response_a": compare_lengths("response_a"),
            "response_b": compare_lengths("response_b"),
        },
    }

    output_file = output_dir / "tie_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def generate_summary_report(
    basic_info, label_dist, text_lengths, model_dist, tie_analysis, output_dir
):
    """Generate a human-readable summary report."""
    lines = []
    lines.append("=" * 60)
    lines.append("Dataset Analysis Report")
    lines.append("=" * 60)
    lines.append("")

    lines.append("1. Basic Information")
    lines.append("-" * 40)
    lines.append(f"Train samples: {basic_info['train_samples']:,}")
    lines.append(f"Test samples: {basic_info['test_samples']:,}")
    lines.append("")

    lines.append("2. Label Distribution")
    lines.append("-" * 40)
    for label, ratio in label_dist["label_ratios"].items():
        count = label_dist["label_counts"][label]
        lines.append(f"{label}: {count:,} ({ratio:.2%})")
    lines.append("")

    lines.append("3. Text Length Statistics (Training Set)")
    lines.append("-" * 40)
    for col, stats in text_lengths["train"].items():
        lines.append(f"{col}:")
        lines.append(f"  Mean: {stats['mean']:,.0f}, Median: {stats['median']:,.0f}")
        lines.append(f"  Range: [{stats['min']:,}, {stats['max']:,}]")
        lines.append(f"  P95: {stats['p95']:,.0f}, P99: {stats['p99']:,.0f}")
    lines.append("")

    lines.append("4. Model Information")
    lines.append("-" * 40)
    lines.append(f"Unique models: {model_dist['unique_models']}")
    lines.append("")

    lines.append("5. Tie Sample Analysis")
    lines.append("-" * 40)
    lines.append(f"Tie samples: {tie_analysis['tie_sample_count']:,}")
    lines.append(f"Tie ratio: {tie_analysis['tie_ratio']:.2%}")
    lines.append("")

    lines.append("=" * 60)

    report_text = "\n".join(lines)
    output_file = output_dir / "summary_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def main():
    base_dir = Path(__file__).parent.parent
    train_path = base_dir / "kaggle" / "input" / "competitions" / "llm-classification-finetuning" / "train.csv"
    test_path = base_dir / "kaggle" / "input" / "competitions" / "llm-classification-finetuning" / "test.csv"
    output_dir = base_dir / "outputs" / "data_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df, test_df = load_data(train_path, test_path)

    print("Analyzing basic info...")
    basic_info = analyze_basic_info(train_df, test_df, output_dir)

    print("Analyzing labels...")
    label_dist = analyze_labels(train_df, output_dir)

    print("Analyzing text lengths...")
    text_lengths = analyze_text_lengths(train_df, test_df, output_dir)

    print("Analyzing models...")
    model_dist = analyze_models(train_df, output_dir)

    print("Analyzing tie samples...")
    tie_analysis = analyze_tie_samples(train_df, output_dir)

    print("Generating summary report...")
    summary = generate_summary_report(
        basic_info, label_dist, text_lengths, model_dist, tie_analysis, output_dir
    )

    print(f"\nAnalysis complete. Results saved to: {output_dir}")
    print("\n" + summary)


if __name__ == "__main__":
    main()
