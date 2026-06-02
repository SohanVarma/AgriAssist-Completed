from pathlib import Path
import json
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_JSON = RESULTS_DIR / "backbone_comparison_web.json"
OUTPUT_MD = RESULTS_DIR / "backbone_comparison_report.md"


def load_existing_metrics():
    possible_files = [
        RESULTS_DIR / "backbone_comparison.json",
        RESULTS_DIR / "model_backbone_comparison.json",
        RESULTS_DIR / "backbone_results.json",
    ]

    for path in possible_files:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass

    return None


def build_default_report():
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "title": "Backbone Comparison: ResNet18 vs Vision Transformer",
        "models": [
            {
                "name": "ResNet18",
                "type": "Convolutional Neural Network",
                "strengths": [
                    "Strong image feature extraction for leaf texture, edges, spots, and disease patterns",
                    "Works well with smaller agricultural image datasets",
                    "Faster and lighter than large transformer models",
                    "Good baseline model for crop disease classification"
                ],
                "limitations": [
                    "May miss long-range image dependencies",
                    "Performance depends strongly on dataset quality and augmentation"
                ],
                "expected_use": "Recommended as the practical baseline model for this AgriAssist prototype."
            },
            {
                "name": "Vision Transformer ViT-B16",
                "type": "Transformer-based image model",
                "strengths": [
                    "Can capture broader global relationships in an image",
                    "Useful when large-scale training data is available",
                    "Can perform strongly with transfer learning"
                ],
                "limitations": [
                    "Usually requires more data and compute",
                    "May overfit or underperform on small datasets",
                    "Slower and heavier for lightweight deployment"
                ],
                "expected_use": "Useful for experimentation and comparison, but less deployment-friendly for this prototype."
            }
        ],
        "comparison_summary": {
            "accuracy": "ResNet18 is usually more stable for small or medium crop disease datasets, while ViT can perform better when enough data and compute are available.",
            "speed": "ResNet18 is faster and lighter for local or web deployment.",
            "interpretability": "CNN backbones such as ResNet are easier to connect with Grad-CAM heatmaps.",
            "deployment": "ResNet18 is more suitable for the current FastAPI website deployment.",
            "recommendation": "Use CNN/ResNet-style models for the main deployed classifier and keep ViT as a research comparison."
        }
    }

    return report


def format_report_as_markdown(report):
    lines = []

    lines.append("# Backbone Comparison Report")
    lines.append("")
    lines.append(f"Generated at: {report.get('generated_at', 'unknown')}")
    lines.append("")
    lines.append("## Compared Models")
    lines.append("")

    for model in report.get("models", []):
        lines.append(f"### {model.get('name', 'Unknown Model')}")
        lines.append("")
        lines.append(f"**Type:** {model.get('type', 'unknown')}")
        lines.append("")
        lines.append("**Strengths:**")
        for item in model.get("strengths", []):
            lines.append(f"- {item}")
        lines.append("")
        lines.append("**Limitations:**")
        for item in model.get("limitations", []):
            lines.append(f"- {item}")
        lines.append("")
        lines.append(f"**Use in AgriAssist:** {model.get('expected_use', '')}")
        lines.append("")

    summary = report.get("comparison_summary", {})

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Accuracy:** {summary.get('accuracy', '')}")
    lines.append(f"- **Speed:** {summary.get('speed', '')}")
    lines.append(f"- **Interpretability:** {summary.get('interpretability', '')}")
    lines.append(f"- **Deployment:** {summary.get('deployment', '')}")
    lines.append(f"- **Recommendation:** {summary.get('recommendation', '')}")
    lines.append("")

    return "\n".join(lines)


def generate_backbone_comparison_report():
    existing_metrics = load_existing_metrics()

    if existing_metrics:
        report = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": "Backbone Comparison Report",
            "raw_metrics": existing_metrics,
            "models": [
                {
                    "name": "ResNet18",
                    "type": "Convolutional Neural Network",
                    "strengths": [
                        "Efficient convolutional feature extraction",
                        "Good practical baseline for crop disease classification",
                        "Compatible with Grad-CAM explainability"
                    ],
                    "limitations": [
                        "May capture less global context than transformer backbones"
                    ],
                    "expected_use": "Recommended for deployed prototype."
                },
                {
                    "name": "Vision Transformer ViT-B16",
                    "type": "Transformer-based image classifier",
                    "strengths": [
                        "Captures global visual relationships",
                        "Useful for research comparison"
                    ],
                    "limitations": [
                        "Requires more data and compute",
                        "Less lightweight for deployment"
                    ],
                    "expected_use": "Recommended for research comparison."
                }
            ],
            "comparison_summary": {
                "accuracy": "Use the stored raw metrics to compare validation accuracy.",
                "speed": "ResNet18 is generally faster and lighter.",
                "interpretability": "ResNet18/CNN features are more directly suited for Grad-CAM.",
                "deployment": "ResNet18 is more deployment-friendly.",
                "recommendation": "Use ResNet/CNN for the website classifier and ViT for experimental comparison."
            }
        }
    else:
        report = build_default_report()

    markdown = format_report_as_markdown(report)

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    OUTPUT_MD.write_text(markdown, encoding="utf-8")

    return {
        "report": report,
        "markdown": markdown,
        "json_path": str(OUTPUT_JSON),
        "markdown_path": str(OUTPUT_MD),
    }


if __name__ == "__main__":
    result = generate_backbone_comparison_report()
    print(result["markdown"])
