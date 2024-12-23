#
# This lib include tools for the 7th Step of the Capstone Project
#

import matplotlib.pyplot as plt

def plot_loss_and_metrics(train_losses, valid_losses, test_losses=None):

    # Plot metrics after training
    plt.figure(figsize=(12, 3))

    # Plot training/validation losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(valid_losses, label="Validation Loss")
    if test_losses != None:
        plt.plot(test_losses, label="Test Loss")
    plt.title("Losses over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot IoU for validation and test sets
    plt.subplot(1, 2, 2)
    plt.plot(train_ious, label="Training IoU")
    plt.plot(valid_ious, label="Validation IoU")
    if test_losses != None:
        plt.plot(test_ious, label="Test IoU")
    plt.title("IoU over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("IoU")
    plt.legend()

    plt.show()