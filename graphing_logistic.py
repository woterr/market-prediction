import matplotlib.pyplot as plt
import numpy as np
def plot_predictions(true_labels, predicted_probs, threshold=0.45, dates=None):
    predicted_labels = (predicted_probs >= threshold).astype(int)

    plt.figure(figsize=(15, 7))

    x = range(len(true_labels))

    # Plot true labels as black dashed line with circle markers
    plt.plot(x, true_labels, 'k--o', label='True Labels', markersize=7, linewidth=2, alpha=0.8)

    # Plot predicted probabilities as orange solid line
    plt.plot(x, predicted_probs, 'orange', label='Predicted Probability', linewidth=2)

    # Plot predicted labels as large colored squares (green=Up, red=Down)
    for i, pred in enumerate(predicted_labels):
        color = 'green'
        plt.scatter(i, pred, color=color, s=150, marker='s', edgecolors='black', label='Predicted Label' if i == 0 else "", alpha=0.9)

    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['Down', 'Up'])
    plt.xlabel('Day Index' if dates is None else 'Date', fontsize=14)
    plt.ylabel('Stock Movement', fontsize=14)
    plt.title(f'Stock Market Prediction vs True Movement (Threshold = {threshold})', fontsize=16)

    # Avoid duplicate legend entries for predicted labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)

    if dates is not None:
        plt.xticks(ticks=x, labels=dates, rotation=45, fontsize=10)
    else:
        plt.xticks(fontsize=12)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()