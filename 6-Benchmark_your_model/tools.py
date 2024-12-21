#
# This library groups tools for the capstone project
#

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import torch
import cv2
import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# def compute_confusion_matrix_with_distributions(model, test_loader, device):
#     """
#     Compute the confusion matrix for all pixels in the test set across all batches.
#     Args:
#         model: Trained segmentation model.
#         test_loader: DataLoader for the test set.
#         device: Device ('cuda' or 'cpu').
#
#     Returns:
#         cm: Confusion matrix (2x2 for binary classification).
#         pos_probs: Predicted probabilities for the positive class (PV).
#         neg_probs: Predicted probabilities for the negative class (Background).
#         nb_of_images_eval: Total number of images evaluated.
#     """
#     model.eval()  # Set model to evaluation mode
#     all_preds = []
#     all_masks = []
#
#     nb_of_images_eval = 0
#
#     # Debug: Check the dataset size
#     print(f"Test loader dataset size: {len(test_loader.dataset)}")
#     print(f"Batch size: {test_loader.batch_size}, Total batches: {len(test_loader)}")
#
#     with torch.no_grad():
#         for batch_idx, (images, masks) in enumerate(test_loader):
#             # Ensure the batch contains images
#             if len(images) == 0:
#                 print(f"Skipped empty batch {batch_idx + 1}/{len(test_loader)}.")
#                 continue
#
#             # Debug: Print batch size
#             print(f"Processing batch {batch_idx + 1}/{len(test_loader)}: {len(images)} images.")
#
#             # Move images and masks to the device
#             images = images.to(device)
#             masks = masks.to(device)
#
#             # Get predictions
#             preds = model(images)
#
#             # Ensure predictions and masks have consistent shapes
#             if preds.shape != masks.shape:
#                 print(f"Shape mismatch in batch {batch_idx + 1}: preds {preds.shape}, masks {masks.shape}")
#                 continue  # Skip batch with inconsistent shapes
#
#             # Debug: Print shapes of predictions and masks
#             print(f"Batch {batch_idx + 1} shapes - preds: {preds.shape}, masks: {masks.shape}")
#
#             # Flatten predictions and masks for confusion matrix computation
#             preds_flat = preds.cpu().numpy().ravel()
#             masks_flat = masks.cpu().numpy().ravel()
#
#             # Debug: Print flattened shapes
#             print(f"Flattened batch {batch_idx + 1} shapes - preds: {preds_flat.shape}, masks: {masks_flat.shape}")
#
#             # Append the flattened predictions and masks
#             all_preds.append(preds_flat)
#             all_masks.append(masks_flat)
#
#             nb_of_images_eval += len(images)
#
#     # Concatenate all predictions and masks across all batches
#     all_preds = np.concatenate(all_preds)
#     all_masks = np.concatenate(all_masks)
#
#     # Debug: Print final concatenated shapes
#     print(f"Final concatenated shapes - all_preds: {all_preds.shape}, all_masks: {all_masks.shape}")
#     assert len(all_preds) == len(all_masks), (
#         f"Length mismatch after processing all batches: preds {len(all_preds)}, masks {len(all_masks)}"
#     )
#
#     # Convert probabilities to binary predictions (threshold = 0.5)
#     binary_preds = (all_preds > 0.5).astype(int)
#
#     # Compute confusion matrix
#     cm = confusion_matrix(all_masks, binary_preds, labels=[0, 1])
#
#     # Separate probabilities for positive and negative classes
#     pos_probs = all_preds[all_masks == 1]  # Positive class (PV)
#     neg_probs = all_preds[all_masks == 0]  # Negative class (Background)
#
#     return cm, pos_probs, neg_probs, nb_of_images_eval

def compute_confusion_matrix_with_distributions(model, test_loader, device):
    model.eval()
    all_preds = []
    all_masks = []
    nb_of_images_eval = 0

    print(f"Test loader dataset size: {len(test_loader.dataset)}")
    print(f"Batch size: {test_loader.batch_size}, Total batches: {len(test_loader)}")

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            if len(images) == 0:
                print(f"Skipped empty batch {batch_idx + 1}/{len(test_loader)}.")
                continue

            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)

            # Align the shape of preds with masks
            preds = preds.squeeze(1)  # Remove channel dimension if present
            masks = masks.unsqueeze(1) if masks.ndim == 3 else masks

            preds = preds.view(preds.size(0), preds.size(1), preds.size(2))
            masks = masks.view(masks.size(0), masks.size(1), masks.size(2))

            if preds.shape != masks.shape:
                print(f"Shape mismatch in batch {batch_idx + 1}: preds {preds.shape}, masks {masks.shape}")
                continue

            all_preds.append(preds.cpu().numpy().ravel())
            all_masks.append(masks.cpu().numpy().ravel())
            nb_of_images_eval += len(images)

    if not all_preds or not all_masks:
        raise ValueError("No valid predictions or masks were generated. Check your model and dataset.")

    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)

    print(f"Final concatenated shapes - all_preds: {all_preds.shape}, all_masks: {all_masks.shape}")
    assert len(all_preds) == len(all_masks), "Prediction and mask lengths do not match."

    binary_preds = (all_preds > 0.5).astype(int)
    cm = confusion_matrix(all_masks, binary_preds, labels=[0, 1])

    pos_probs = all_preds[all_masks == 1]
    neg_probs = all_preds[all_masks == 0]

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


class CustomDataset(data.Dataset):
    def __init__(self, image_paths, target_paths, transform=None, band=None, device='cuda'):

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform
        self.band = band  # Specify which band to use (0: R, 1: G, 2: B, 3: NIR, None: all bands)
        self.scaler = MinMaxScaler()
        self.device = device  # Specify the device for preloading

        # Preload images and masks into GPU memory
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Preloading images and masks into GPU memory...")
        # self.images = [torch.tensor(cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32), device=self.device) for path in image_paths]
        # self.masks = [torch.tensor(cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32), device=self.device) for path in target_paths]
        self.images = image_paths
        self.masks = target_paths
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Preloading complete.")

    def __getitem__(self, index):
        # Debugging: Print dataset lengths and current index
        print(f"[DEBUG] Dataset size: {len(self.image_paths)} images, {len(self.masks)} masks")
        print(f"[DEBUG] Current index: {index}")

        # Access the paths safely
        try:
            # Get the current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Log the index being loaded
            print(f"[{timestamp}] Loading image and mask for index {index}...")

            # Retrieve preloaded image and mask
            # image = self.images[index]
            # mask = self.masks[index]
            # Debugging: Add print statements for image and mask paths
            print(f"Loading image: {self.image_paths[index]}")
            print(f"Loading mask: {self.masks[index]}")
            image = torch.tensor(cv2.imread(self.images[index], cv2.IMREAD_UNCHANGED).astype(np.float32))
            mask = torch.tensor(cv2.imread(self.masks[index], cv2.IMREAD_UNCHANGED).astype(np.float32))
            print(f"Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")

            # Select a specific band if specified
            if self.band is not None:
                image = image[:, :, self.band].unsqueeze(2)  # Add channel dimension

            # Normalize the image
            # image_reshaped = image.reshape(-1, image.shape[-1])
            # image_scaled = self.scaler.fit_transform(image_reshaped)
            # image = image_scaled.reshape(image.shape)

            # Normalize the image (Min-Max normalization)
            image_min = image.min()
            image_max = image.max()
            if image_max > image_min:  # Avoid division by zero
                image = (image - image_min) / (image_max - image_min)

            # Reshape for MinMaxScaler and apply normalization
            # image_reshaped = image.reshape(-1, 4)
            # image_scaled = self.scaler.fit_transform(image_reshaped)
            # image = image_scaled.reshape(image.shape)

            # Load the 1-band binary mask
            # mask = imageio.imread(self.target_paths[index])
            # mask = np.asarray(mask, dtype='float32')
            # mask = np.where(mask>1, 0, mask) # some images has soil annotations as well
            mask = mask.to(self.device)
            mask = torch.where(mask > 1, torch.tensor(0.0, device=self.device), mask)

            # Debugging: Print shapes and types
            # print(f"Image shape: {image.shape}, dtype: {image.dtype}")
            # print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")

            # Debugging: Log the properties of the loaded image and mask
            print(f"[{timestamp}] Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"[{timestamp}] Mask shape: {mask.shape}, dtype: {mask.dtype}")

            # Apply the transformation to both image and mask if self.transform is set

            if self.transform:
                try:
                    if not isinstance(image, torch.Tensor):
                        image = self.transform(image)  # Apply transform only if not already a tensor
                    mask = torch.tensor(mask, dtype=torch.float32)  # Ensure mask is a tensor
                except Exception as e:
                    print(f"[{timestamp}] Error during transformation at index {index}: {e}")
                    raise e

            return image, mask

        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error at index {index}: {e}")
            raise e

    def __len__(self):

        return len(self.image_paths)

