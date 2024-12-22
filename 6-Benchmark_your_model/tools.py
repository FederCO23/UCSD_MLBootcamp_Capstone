#
# This library groups tools for the capstone project
#
import imageio
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, jaccard_score

def evaluate_random_model(test_mask_paths, random_preds):
    """
    Evaluate the Random Model using the confusion matrix.
    """
    all_preds = []
    all_masks = []

    for mask_path, pred in zip(test_mask_paths, random_preds):
        mask = imageio.imread(mask_path).flatten()
        pred = pred.flatten()
        all_masks.extend(mask)
        all_preds.extend(pred)

    cm = confusion_matrix(all_masks, all_preds, labels=[0, 1])
    return cm, np.array(all_masks), np.array(all_preds)


def plot_confusion_matrix(cm):
    """
    Plot the confusion matrix using sklearn's ConfusionMatrixDisplay.
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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, colorbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()


def plot_NIR_hist_per_class(image_paths, mask_paths, num_samples=5, thresholds=np.arange(0, 10000, 50), metric="f1"):
    """
    Plot histograms for NIR band pixel intensities for positive and negative classes and compute the optimal threshold.
    """

    nir_values_positive = []
    nir_values_negative = []

    for img_path, mask_path in zip(image_paths[:num_samples], mask_paths[:num_samples]):
        image = imageio.imread(img_path)
        mask = imageio.imread(mask_path)

        # define the two arrays
        nir_values = image[:, :, 3].flatten()
        mask_values = mask[:, :].flatten()

        # multiply element-wise the image array with the mask array.
        # Then, the result are the reflectance pixel value for the NIR band where the mask is positive (1)
        nir_values_positive.extend(nir_values * mask_values)
        nir_values_negative.extend(nir_values * (1 - mask_values))

    # drop the zero values
    # (consider that the minimum values for the NIR band images across the entire dataset are greater than zero)
    nir_values_positive = np.array([nir_values_positive])
    nir_values_negative = np.array([nir_values_negative])
    positives_nonzero = nir_values_positive[nir_values_positive != 0]
    negatives_nonzero = nir_values_negative[nir_values_negative != 0]

    # Find the optimal threshold using the helper function which maximize the selected metric
    optimal_threshold, best_score = find_optimal_threshold_from_histogram(
        positives_nonzero, negatives_nonzero, thresholds=thresholds, metric=metric
    )

    plt.figure(figsize=(10, 6))
    plt.hist(positives_nonzero, bins=200, color='red', alpha=0.7, label="Positive class NIR pixels distribution")
    plt.hist(negatives_nonzero, bins=200, color='grey', alpha=0.7, label="Negative class NIR pixels distribution")
    plt.axvline(optimal_threshold, color='blue', linestyle='--', label=f"Optimal Threshold: {optimal_threshold:.2f}")
    plt.title("NIR Band Pixel Distribution for Positive and Negative Mask Classes")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()


    nb_pixels = positives_nonzero.shape[0] + negatives_nonzero.shape[0]
    nb_images = nb_pixels / (image.shape[0] * image.shape[1])

    print(f"Optimal Threshold: {optimal_threshold:.2f}")
    print(f"Best {metric.capitalize()} Score: {best_score:.4f}")
    print(f"{nb_pixels:,} pixels considered from {int(nb_images)} images")

    return optimal_threshold


def find_optimal_threshold_from_histogram(nir_values_positive, nir_values_negative, thresholds=np.arange(0, 10000, 50),
                                          metric="f1"):
    """
    Find the optimal threshold based on the NIR band pixel distributions and maximize a chosen metric.

    """
    best_score = 0
    best_threshold = 0

    for threshold in thresholds:
        # Generate predictions based on the threshold
        preds_positive = (nir_values_positive < threshold).astype(int)
        preds_negative = (nir_values_negative < threshold).astype(int)

        # True labels (1 for positives, 0 for negatives)
        true_positive_labels = np.ones_like(preds_positive)
        true_negative_labels = np.zeros_like(preds_negative)

        # Concatenate predictions and labels
        preds = np.concatenate([preds_positive, preds_negative])
        true_labels = np.concatenate([true_positive_labels, true_negative_labels])

        # Compute the selected metric
        if metric == "f1":
            score = f1_score(true_labels, preds)
        elif metric == "iou":
            score = jaccard_score(true_labels, preds)
        else:
            raise ValueError("Unsupported metric. Choose 'f1' or 'iou'.")

        # Update the best score and threshold
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
