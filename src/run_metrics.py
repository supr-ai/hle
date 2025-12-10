import os
import json
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt


# source: https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py
def calib_err(confidence, correct, p="2", beta=100):
    """Calculate calibration error."""
    if len(confidence) < beta:
        beta = max(1, len(confidence) // 2)

    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    if not bins:
        return 0.0
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0] : bins[i][1]]
        bin_correct = correct[bins[i][0] : bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == "2":
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p == "infty" or p == "infinity" or p == "max":
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == "2":
        cerr = np.sqrt(cerr)

    return cerr


def compute_metrics(predictions):
    """Compute metrics for each model from judged predictions."""
    model_results = {}  # model_key -> {"correct": [], "confidence": []}

    for question_id, v in predictions.items():
        if "judge_response" not in v:
            continue

        judge_response = v["judge_response"]
        for model_key, model_judge in judge_response.items():
            if model_key not in model_results:
                model_results[model_key] = {"correct": [], "confidence": []}

            model_results[model_key]["correct"].append("yes" in model_judge["correct"])
            model_results[model_key]["confidence"].append(model_judge["confidence"])

    metrics = {}
    for model_key, data in model_results.items():
        correct = np.array(data["correct"])
        confidence = np.array(data["confidence"]) / 100
        n = len(correct)

        if n == 0:
            continue

        accuracy = sum(correct) / n
        # Variance of binomial proportion
        variance = accuracy * (1 - accuracy) / n

        # Calibration error (only meaningful with enough samples)
        if n >= 10:
            calibration_error = calib_err(
                confidence, correct, p="2", beta=min(100, n // 2)
            )
        else:
            calibration_error = None

        metrics[model_key] = {
            "accuracy": round(accuracy * 100, 2),
            "variance": round(variance * 10000, 4),  # as percentage points squared
            "n": n,
            "calibration_error": (
                round(calibration_error * 100, 2)
                if calibration_error is not None
                else None
            ),
        }

    return metrics


def create_bar_chart(metrics, output_path):
    """Create a bar chart of accuracy % for each model."""
    # Sort by accuracy descending
    sorted_models = sorted(
        metrics.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    # Prepare data for plotting
    labels = []
    accuracies = []
    ns = []

    for model_key, data in sorted_models:
        # Replace "main" with "Sup AI"
        display_name = "Sup AI" if model_key == "main" else model_key
        labels.append(f"{display_name}\n(n={data['n']})")
        accuracies.append(data["accuracy"])
        ns.append(data["n"])

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.8), 8))

    # Create bars
    bars = ax.bar(
        range(len(labels)),
        accuracies,
        color="#4A90D9",
        edgecolor="black",
        linewidth=0.5,
    )

    # Highlight "Sup AI" bar
    for i, (model_key, _) in enumerate(sorted_models):
        if model_key == "main":
            bars[i].set_color("#E74C3C")
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(1.5)

    # Add value labels on bars
    for i, (acc, n) in enumerate(zip(accuracies, ns)):
        ax.text(
            i,
            acc + 0.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Customize plot
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Model Accuracy on HLE Benchmark", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(accuracies) + 10)
    ax.grid(axis="y", alpha=0.3)

    # Add a legend for Sup AI highlighting
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#E74C3C", edgecolor="black", label="Sup AI"),
        Patch(facecolor="#4A90D9", edgecolor="black", label="Other Models"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved bar chart to {output_path}")


def create_questions_table(predictions, output_path):
    """Create a markdown table showing per-question correctness for each model."""
    # Collect all model keys and build per-question results
    all_models = set()
    question_results = (
        {}
    )  # question_id -> {model_key -> "correct"/"incorrect"/"no_answer"}

    for question_id, v in predictions.items():
        question_results[question_id] = {}
        if "judge_response" not in v:
            continue

        judge_response = v["judge_response"]
        for model_key, model_judge in judge_response.items():
            all_models.add(model_key)
            if "yes" in model_judge["correct"]:
                question_results[question_id][model_key] = "correct"
            else:
                question_results[question_id][model_key] = "incorrect"

    # Sort models: "main" (Sup AI) first, then alphabetically
    sorted_models = sorted(all_models, key=lambda x: (x != "main", x))

    # Sort questions by ID for consistent ordering
    sorted_questions = sorted(question_results.keys())

    # Build markdown table
    lines = []

    # Header row
    header_cols = ["Question"]
    for model_key in sorted_models:
        display_name = "Sup AI" if model_key == "main" else model_key
        header_cols.append(display_name)
    lines.append("| " + " | ".join(header_cols) + " |")

    # Separator row
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    # Data rows
    for question_id in sorted_questions:
        row_cols = [question_id]
        for model_key in sorted_models:
            result = question_results[question_id].get(model_key)
            if result == "correct":
                row_cols.append("✅")
            elif result == "incorrect":
                row_cols.append("❌")
            else:
                row_cols.append("⚪")  # Gray circle for no answer
        lines.append("| " + " | ".join(row_cols) + " |")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved questions table to {output_path}")


def main(args):
    # Load judged predictions
    with open(args.predictions, "r") as f:
        predictions = json.load(f)

    # Compute metrics
    metrics = compute_metrics(predictions)

    # Determine output directory (base dir of repo)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Save metrics.json
    metrics_json_path = os.path.join(base_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_json_path}")

    # Print metrics summary
    print("\n*** Metrics by Model ***")
    for model_key in sorted(
        metrics.keys(), key=lambda k: metrics[k]["accuracy"], reverse=True
    ):
        data = metrics[model_key]
        display_name = "Sup AI" if model_key == "main" else model_key
        calib_str = (
            f" | ECE={data['calibration_error']}%"
            if data["calibration_error"] is not None
            else ""
        )
        print(
            f"{display_name}: {data['accuracy']}% | n={data['n']} | var={data['variance']}{calib_str}"
        )

    # Create bar chart
    chart_path = os.path.join(base_dir, "metrics.png")
    create_bar_chart(metrics, chart_path)

    # Create questions markdown table
    questions_path = os.path.join(base_dir, "questions.md")
    create_questions_table(predictions, questions_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate metrics and visualizations from judged predictions"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="judged_hle_pro.json",
        help="Path to judged predictions JSON file",
    )
    args = parser.parse_args()
    main(args)
