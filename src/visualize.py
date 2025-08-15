import matplotlib.pyplot as plt


def plot_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(results["train_loss"], label="Train Loss")
    axes[0].plot(results["val_loss"], label="Val Loss")
    axes[1].plot(results["train_acc"], label="Train Accuracy")
    axes[1].plot(results["val_acc"], label="Val Accuracy")

    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy")

    for i in range(2):
        axes[i].grid(True)
        axes[i].set_xlabel("Epoch")
        axes[i].legend()

    fig.suptitle("Evaluation throughout epochs", fontsize=16)
    plt.tight_layout()
    plt.show()
