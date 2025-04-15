import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def sort_and_get_pred(predictions, df, Y):
    sorted_confidence, sorted_confidence_indices = predictions.sort(descending=True)

    predicted_class = sorted_confidence_indices[:, 0]
    predicted_confidence = sorted_confidence[:, 0]

    true_values = df[Y].values
    well_predicted = (predicted_class == true_values).float()

    return sorted_confidence, well_predicted, predicted_confidence, predicted_class, true_values


def confidence_histogram(sorted_confidence, well_predicted, df):
    confidence_score = sorted_confidence[:, 0] - sorted_confidence[:, 1]

    # Convert NumPy arrays to a DataFrame
    df = pd.DataFrame(
        {
            "confidence_score": confidence_score.numpy(),
            "well_predicted": well_predicted.numpy(),  # Ensure this is categorical if needed
        }
    )

    # Plot with proper data format
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x="confidence_score", bins=100, hue="well_predicted", stat="percent")

    return fig


def calibration_curve(n_bins, confidences, predicted_classes, true_labels):
    # Initialize bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)  # Bins from 0 to 1
    bin_accs = []
    bin_confidences = []

    # Compute accuracy per bin
    for i in range(n_bins):
        bin_mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if bin_mask.sum() > 5:
            bin_acc = (predicted_classes[bin_mask] == true_labels[bin_mask]).float().mean()
            bin_conf = confidences[bin_mask].mean()
            bin_accs.append(bin_acc.item())
            bin_confidences.append(bin_conf.item())

    # Convert to numpy
    bin_confidences = np.array(bin_confidences)
    bin_accs = np.array(bin_accs)

    # Plot reliability diagram
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect Calibration")
    plt.plot(bin_confidences, bin_accs, marker="o", color="blue", label="Model Calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid()
    return fig


def get_automatic_accuracy(thresholds, predicted_confidence, predicted_class, true_values):
    """
    We automatically code the APE if the confidence is above the threshold.
    Compute the accuracy on the automatically coded APEs.

    Args:
        thresholds (np.array(float), shape (n_thresholds,)): The threshold for automatic coding.
        predicted_confidence (torch.Tensor, shape (n_samples, 1): The confidence of the predictions.
        predicted_class (torch.Tensor, shape (n_samples, 1)
            The predicted class of the APEs.
        true_values (torch.Tensor, shape (n_samples, 1): The true values of the APEs.

    Returns:
        automatic_coding_rate (float): The rate of automatically coded APEs.
        accuracy_automatic (float): The accuracy on the automatically coded APEs.

    """
    n_thresholds = len(thresholds)
    automatic_coding_mask = predicted_confidence[:, None] >= thresholds[None, :]
    automatic_coding_rate = automatic_coding_mask.mean(axis=0)

    predicted_class_expanded = np.repeat(predicted_class[:, None], n_thresholds, 1)
    true_values_expanded = np.repeat(true_values[:, None], n_thresholds, 1)

    predicted_automatic = np.ma.array(predicted_class_expanded, mask=~automatic_coding_mask)
    ground_truth_automatic = np.ma.array(true_values_expanded, mask=~automatic_coding_mask)
    accuracy_automatic = (predicted_automatic == ground_truth_automatic).mean(axis=0)
    return automatic_coding_rate, accuracy_automatic


def plot_automatic_coding_accuracy_curve(torchft_plot, ft_plot, thresholds):
    """
    Plot the automatic coding accuracy curve.
    Args:
        torchft_plot (tuple): The plot data for torchft.
        ft_plot (tuple): The plot data for ft.

        Both of them should be obtained from the get_automatic_accuracy function above.

    Returns:
        fig (plotly.graph_objects.Figure): The Plotly figure object.

    """

    # Create masks for the plots
    mask_torchft = torchft_plot[0] > 0
    mask_ft = ft_plot[0] > 0

    # Create the matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot torchft data
    ax.scatter(torchft_plot[0][mask_torchft], torchft_plot[1][mask_torchft], label="torchft")

    # Plot ft data
    ax.scatter(ft_plot[0][mask_ft], ft_plot[1][mask_ft], label="ft")

    # Set labels and title
    ax.set_xlabel("Pourcentage de codif automatique")
    ax.set_ylabel("Accuracy")

    # Add legend
    ax.legend(loc="upper right", fancybox=True, shadow=True)

    # Set grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    return fig
