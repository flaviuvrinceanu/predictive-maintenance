from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")

RUL_RESULTS_PATH = RESULTS_DIR / "lstm_engine_val_seq30_cap125_predictions.csv"
ANOMALY_RESULTS_PATH = RESULTS_DIR / "lstm_autoencoder_strict_hidden32_l1_detailed_results.csv"
THRESHOLD_RESULTS_PATH = RESULTS_DIR / "lstm_autoencoder_strict_hidden32_l1_threshold_sensitivity.csv"


rul_df = pd.read_csv(RUL_RESULTS_PATH)
anomaly_df = pd.read_csv(ANOMALY_RESULTS_PATH)
threshold_df = pd.read_csv(THRESHOLD_RESULTS_PATH)


SEQ_LENGTH = 30

anomaly_df["end_cycle"] = anomaly_df["cycle"].astype(int)
anomaly_df["start_cycle"] = anomaly_df["end_cycle"] - SEQ_LENGTH + 1

anomaly_df["true_state"] = anomaly_df["label"].map({
    0: "Healthy",
    1: "Anomaly"
})

anomaly_df["predicted_state"] = anomaly_df["predicted_label"].map({
    0: "Healthy",
    1: "Anomaly"
})


anomaly_df = anomaly_df.sort_values(
    by=["unit", "end_cycle"],
    ascending=True
).reset_index(drop=True)


anomaly_df["example_index"] = anomaly_df.index

anomaly_df["example_name"] = (
    "Engine " + anomaly_df["unit"].astype(int).astype(str).str.zfill(3)
    + " | Window " + anomaly_df["start_cycle"].astype(str)
    + "-" + anomaly_df["end_cycle"].astype(str)
    + " | RUL " + anomaly_df["rul"].astype(int).astype(str)
    + " | " + anomaly_df["true_state"]
)





if 95 in threshold_df["percentile"].values:
    FINAL_THRESHOLD = float(
        threshold_df.loc[
            threshold_df["percentile"] == 95,
            "threshold"
        ].iloc[0]
    )
else:
    FINAL_THRESHOLD = float(threshold_df["threshold"].iloc[-1])


def label_to_text(label):
    return "Healthy" if int(label) == 0 else "Anomaly"




def rul_prediction_demo(engine_id):
    engine_id = int(engine_id)

    row = rul_df[rul_df["engine_id"] == engine_id].iloc[0]

    true_rul = float(row["true_rul"])
    predicted_rul = float(row["predicted_rul"])
    error = float(row["error"])
    abs_error = float(row["abs_error"])

    summary = f"""
### RUL Prediction Result

**Selected engine:** {engine_id}

| Metric | Value |
|---|---:|
| True RUL | {true_rul:.2f} |
| Predicted RUL | {predicted_rul:.2f} |
| Error | {error:.2f} |
| Absolute Error | {abs_error:.2f} |

The RUL model estimates the remaining useful life of the selected engine based on the final available time window.
"""

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        rul_df["true_rul"],
        rul_df["predicted_rul"],
        alpha=0.7,
        label="Test engines"
    )

    ax.scatter(
        [true_rul],
        [predicted_rul],
        s=120,
        label="Selected engine"
    )

    min_val = min(rul_df["true_rul"].min(), rul_df["predicted_rul"].min())
    max_val = max(rul_df["true_rul"].max(), rul_df["predicted_rul"].max())

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        label="Ideal prediction"
    )

    ax.set_title("Predicted vs True RUL")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return summary, fig




def anomaly_detection_demo(example_name):
    row = anomaly_df[anomaly_df["example_name"] == example_name].iloc[0]

    example_index = int(row["example_index"])
    unit = int(row["unit"])
    start_cycle = int(row["start_cycle"])
    end_cycle = int(row["end_cycle"])
    rul = float(row["rul"])

    true_label = int(row["label"])
    predicted_label = int(row["predicted_label"])

    reconstruction_error = float(row["reconstruction_error"])
    rul_range = row["rul_range"]

    true_state = label_to_text(true_label)
    verdict = label_to_text(predicted_label)

    summary = f"""
### Anomaly Detection Result

**Selected example:** {example_index}  
**Selected engine:** {unit}  
**Evaluated window:** cycles **{start_cycle}–{end_cycle}**  
**RUL at end of window:** {rul:.0f}  
**RUL range:** {rul_range}

| Metric | Value |
|---|---:|
| Reconstruction error | {reconstruction_error:.6f} |
| Final threshold | {FINAL_THRESHOLD:.6f} |
| True state | {true_state} |
| Model prediction | {verdict} |

Each evaluated example is a **30-cycle window**.  
The model does not classify a single cycle, but the full sensor sequence from cycle **{start_cycle}** to cycle **{end_cycle}**.

A window is classified as anomalous when its reconstruction error is above the selected threshold.
"""

    unit_df = anomaly_df[anomaly_df["unit"] == unit].sort_values("end_cycle")

    healthy_points = unit_df[unit_df["label"] == 0]
    anomaly_points = unit_df[unit_df["label"] == 1]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        healthy_points["end_cycle"],
        healthy_points["reconstruction_error"],
        label="Healthy evaluation windows"
    )

    ax.scatter(
        anomaly_points["end_cycle"],
        anomaly_points["reconstruction_error"],
        label="Severe anomaly windows"
    )

    ax.axhline(
        FINAL_THRESHOLD,
        linestyle="--",
        label=f"Threshold = {FINAL_THRESHOLD:.4f}"
    )

    ax.scatter(
        [end_cycle],
        [reconstruction_error],
        s=140,
        label="Selected window"
    )

    ax.set_title(f"Reconstruction Error for Engine {unit}")
    ax.set_xlabel("Window end cycle")
    ax.set_ylabel("Reconstruction Error")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    threshold_fig, threshold_ax = plt.subplots(figsize=(7, 5))

    threshold_ax.plot(
        threshold_df["percentile"],
        threshold_df["precision"],
        marker="o",
        label="Precision"
    )

    threshold_ax.plot(
        threshold_df["percentile"],
        threshold_df["recall"],
        marker="o",
        label="Recall"
    )

    threshold_ax.plot(
        threshold_df["percentile"],
        threshold_df["f1"],
        marker="o",
        label="F1-score"
    )

    threshold_ax.set_title("Threshold Sensitivity")
    threshold_ax.set_xlabel("Threshold percentile")
    threshold_ax.set_ylabel("Score")
    threshold_ax.grid(alpha=0.3)
    threshold_ax.legend()
    threshold_fig.tight_layout()

    unit_table = unit_df[
        [
            "unit",
            "start_cycle",
            "end_cycle",
            "rul",
            "true_state",
            "reconstruction_error",
            "predicted_state",
            "rul_range"
        ]
    ].copy()

    unit_table = unit_table.rename(columns={
        "unit": "Engine",
        "start_cycle": "Window start",
        "end_cycle": "Window end",
        "rul": "RUL",
        "true_state": "True state",
        "reconstruction_error": "Reconstruction error",
        "predicted_state": "Predicted state",
        "rul_range": "RUL range"
    })

    unit_table = unit_table.reset_index(drop=True)

    return summary, fig, threshold_fig, unit_table



engine_choices = sorted(rul_df["engine_id"].astype(int).unique().tolist())
anomaly_choices = anomaly_df["example_name"].tolist()

with gr.Blocks(title="Predictive Maintenance Demo") as demo:
    gr.Markdown(
        """
# Predictive Maintenance Demo

This demo presents the results of a predictive maintenance project based on the NASA CMAPSS dataset.

The project contains two main components:

1. **Remaining Useful Life prediction** using an LSTM model.
2. **Anomaly detection** using an LSTM autoencoder .


"""
    )

    with gr.Tab("RUL Prediction"):
        gr.Markdown(
            """
## RUL Prediction

Select an engine from the  test set.

The application displays the true RUL, predicted RUL, prediction error, and a comparison plot.
"""
        )

        engine_dropdown = gr.Dropdown(
            choices=[str(e) for e in engine_choices],
            value=str(engine_choices[0]),
            label="Select engine"
        )

        rul_output = gr.Markdown()
        rul_plot = gr.Plot()

        engine_dropdown.change(
            fn=rul_prediction_demo,
            inputs=engine_dropdown,
            outputs=[rul_output, rul_plot]
        )

        demo.load(
            fn=rul_prediction_demo,
            inputs=engine_dropdown,
            outputs=[rul_output, rul_plot]
        )

    with gr.Tab("Anomaly Detection"):
        gr.Markdown(
            """
## Anomaly Detection

Select a 30-cycle window from the anomaly detection test set.

Each example represents a sequence of 30 consecutive cycles from one engine.  
The autoencoder reconstructs the selected window and computes a reconstruction error.  
If this error is above the threshold, the window is classified as anomalous.

Only two regions are included in this evaluation:

- healthy windows with RUL ≥ 120
- severe degradation windows with RUL ≤ 30

The intermediate RUL region was excluded because degradation is gradual and harder to label clearly.
"""
        )

        example_dropdown = gr.Dropdown(
            choices=anomaly_choices,
            value=anomaly_choices[0],
            label="Select evaluated 30-cycle window"
        )

        anomaly_output = gr.Markdown()
        anomaly_plot = gr.Plot()
        threshold_plot = gr.Plot()
        unit_table = gr.Dataframe(label="Evaluated windows for selected engine")

        example_dropdown.change(
            fn=anomaly_detection_demo,
            inputs=example_dropdown,
            outputs=[anomaly_output, anomaly_plot, threshold_plot, unit_table]
        )

        demo.load(
            fn=anomaly_detection_demo,
            inputs=example_dropdown,
            outputs=[anomaly_output, anomaly_plot, threshold_plot, unit_table]
        )


if __name__ == "__main__":
    demo.launch()