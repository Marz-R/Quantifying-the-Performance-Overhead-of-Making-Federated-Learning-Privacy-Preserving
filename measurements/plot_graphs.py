import matplotlib.pyplot as plt

class GraphPlotter:
    def __init__(self):
        pass

    def plot_losses(self, train_loss, val_loss, max_epochs):
        epochs = range(1, max_epochs + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
