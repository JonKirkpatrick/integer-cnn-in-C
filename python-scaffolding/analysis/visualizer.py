from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
import numpy as np

class ConfusionMatrixRecorder:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.preds = []
        self.labels = []

    def update(self, preds, labels):
        self.preds.append(preds.detach().cpu().numpy())
        self.labels.append(labels.detach().cpu().numpy())

    def compute(self):
        y_true = np.concatenate(self.labels)
        y_pred = np.concatenate(self.preds)
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-7)
        return cm_norm
    
class ConfusionMatrixWriter:
    def __init__(self, num_classes, output_dir):
        self.num_classes = num_classes
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.colorbar = None

    def write(self, cm_norm, epoch, acc):
        self.ax.clear()
        im = self.ax.imshow(
            cm_norm,
            cmap="magma",
            interpolation="nearest",
            vmin=0,
            vmax=1
        )

        self.ax.set_xticks(range(self.num_classes))
        self.ax.set_yticks(range(self.num_classes))
        self.ax.set_title(f"Epoch {epoch:03d} | Acc: {acc:.4f}")
        self.ax.set_xlabel("Predicted")
        self.ax.set_ylabel("True")

        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(im, ax=self.ax, label="Probability")

        path = os.path.join(self.output_dir, f"epoch_{epoch:04d}.png")
        self.fig.savefig(path, dpi=150)

    def close(self):
        plt.close(self.fig)