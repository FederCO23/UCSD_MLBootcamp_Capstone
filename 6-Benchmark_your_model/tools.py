#
# This library groups tools for the capstone project
#

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

def compute_confusion_matrix_with_distributions(model, test_loader, device):
    """
    Compute the confusion matrix for all pixels in the test set across all batches.
    Args:
        model: Trained segmentation model.
        test_loader: DataLoader for the test set.
        device: Device ('cuda' or 'cpu').

    Returns:
        cm: Confusion matrix (2x2 for binary classification).
        pos_probs: Predicted probabilities for the positive class (PV).
        neg_probs: Predicted probabilities for the negative class (Background).
        nb_of_images_eval: Total number of images evaluated.
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_masks = []

    nb_of_images_eval = 0

    # Debug: Check the dataset size
    print(f"Test loader dataset size: {len(test_loader.dataset)}")
    print(f"Batch size: {test_loader.batch_size}, Total batches: {len(test_loader)}")

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            # Ensure the batch contains images
            if len(images) == 0:
                print(f"Skipped empty batch {batch_idx + 1}/{len(test_loader)}.")
                continue

            # Debug: Print batch size
            print(f"Processing batch {batch_idx + 1}/{len(test_loader)}: {len(images)} images.")

            # Move images and masks to the device
            images = images.to(device)
            masks = masks.to(device)

            # Get predictions
            preds = model(images)

            # Ensure predictions and masks have consistent shapes
            if preds.shape != masks.shape:
                print(f"Shape mismatch in batch {batch_idx + 1}: preds {preds.shape}, masks {masks.shape}")
                continue  # Skip batch with inconsistent shapes

            # Debug: Print shapes of predictions and masks
            print(f"Batch {batch_idx + 1} shapes - preds: {preds.shape}, masks: {masks.shape}")

            # Flatten predictions and masks for confusion matrix computation
            preds_flat = preds.cpu().numpy().ravel()
            masks_flat = masks.cpu().numpy().ravel()

            # Debug: Print flattened shapes
            print(f"Flattened batch {batch_idx + 1} shapes - preds: {preds_flat.shape}, masks: {masks_flat.shape}")

            # Append the flattened predictions and masks
            all_preds.append(preds_flat)
            all_masks.append(masks_flat)

            nb_of_images_eval += len(images)

    # Concatenate all predictions and masks across all batches
    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)

    # Debug: Print final concatenated shapes
    print(f"Final concatenated shapes - all_preds: {all_preds.shape}, all_masks: {all_masks.shape}")
    assert len(all_preds) == len(all_masks), (
        f"Length mismatch after processing all batches: preds {len(all_preds)}, masks {len(all_masks)}"
    )

    # Convert probabilities to binary predictions (threshold = 0.5)
    binary_preds = (all_preds > 0.5).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(all_masks, binary_preds, labels=[0, 1])

    # Separate probabilities for positive and negative classes
    pos_probs = all_preds[all_masks == 1]  # Positive class (PV)
    neg_probs = all_preds[all_masks == 0]  # Negative class (Background)

    return cm, pos_probs, neg_probs, nb_of_images_eval


def display_confusion_matrix_with_metrics_and_distributions(cm, pos_probs, neg_probs, nb_of_images_eval):
    """
    Display the confusion matrix with the traditional layout and print additional metrics.
    Args:
        cm: Confusion matrix (2x2 for binary classification).
    """

    # New labels for the classes
    labels = ["Background", "PV"]

    # Extract values from confusion matrix
    TN = cm[0, 0]  # True Negative
    FP = cm[0, 1]  # False Positive
    FN = cm[1, 0]  # False Negative
    TP = cm[1, 1]  # True Positive

    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity (Recall)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # True Negative Rate
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # Positive Predictive Value
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    IoU = TP / (TP + FN + FP) if (TP + FN + FP) > 0 else 0  # IoU

    # Print metrics
    print(70 * '-')
    print(f"TP:{TP}\tTN:{TN}\tFP:{FP}\t\tFN:{FN}")
    print(70 * '-')
    print("\n")
    print(f"Accuracy:     {accuracy:.4f}\t               (TP + TN) / (TP + TN + FP + FN)")
    print(f"Recall:       {recall:.4f}\t                      TP / (TP + FN)")
    print(f"Specificity:  {specificity:.4f}\t                      TN / (TN + FP)")
    print(f"Precision:    {precision:.4f}\t                      TP / (TP + FP)")
    print(f"F1-Score:     {f1_score:.4f}\t    (2*precision*recall) / (precision + recall)")
    print(f"IoU:          {IoU:.4f}\t                      TP / (TP + FN + FP)")
    print("\n\n")

    # Display the confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")

    # Plot distributions of predicted probabilities
    sns.kdeplot(neg_probs, fill=True, color="green", alpha=0.5, label="Negative (Background)", ax=axes[1])
    sns.kdeplot(pos_probs, fill=True, color="blue", alpha=0.5, label="Positive (PV)", ax=axes[1])

    # Add a threshold line
    axes[1].axvline(0.5, color="black", linestyle="--", label="Threshold")

    # Set titles and labels
    axes[1].set_title("Predicted Probability Distributions")
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_ylabel("Density")

    # Add legend
    axes[1].legend()

