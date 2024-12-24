#
# This lib include tools for the 7th Step of the Capstone Project
#

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
from torch.amp import autocast, GradScaler
from torchmetrics import JaccardIndex
import seaborn as sns
from sklearn.metrics import jaccard_score

def plot_loss_and_metrics(train_losses, train_ious, valid_losses, valid_ious, test_losses=None, test_ious=None):
    """
    Plot the loss function and the IoU metric over epochs for the datasets
    """

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
    if test_ious != None:
        plt.plot(test_ious, label="Test IoU")
    plt.title("IoU over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("IoU")
    plt.legend()

    plt.show()
    
    
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
    print(70*'-')
    print(f"TP:{TP}\tTN:{TN}\tFP:{FP}\t\tFN:{FN}")
    print(70*'-')
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
    

def train_loop(model, train_loader, valid_loader, test_loader, optimizer, loss, scheduler, early_stopping_patience=10, early_stopping_min_delta=0.0001, model_filename='./models/default.pth', device='cpu', ):

    train_losses = []
    valid_losses = []
    test_losses = []
    train_ious = []
    valid_ious = []
    test_ious = []
        
    # Force to include the test set loss and score metric into the training loop
    include_test=False

    # Save the model's params: 
    #model_filename='./models/unet_effb7.pth'

    max_score = float('inf')  # Initialize with a high value to store the best validation loss

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta) 

    # Ensure Model is on the Correct Device
    model = model.to(device)

    scaler = GradScaler('cuda')
    
    tic = time.time()
    for epoch in range(0, 200):
        
        print(f'\nEpoch: {epoch}')
        
        # Run training
        train_loss, train_iou_score = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn=loss, device=device, discard_allbkgnd = True)
        train_losses.append(train_loss)
        train_ious.append(train_iou_score)
        
        
        # Run validation
        valid_loss, valid_iou_score = validate_one_epoch(model, valid_loader, loss_fn=loss, device=device, discard_allbkgnd=True)
        valid_losses.append(valid_loss)
        valid_ious.append(valid_iou_score)

        # Check early stopping criteria with the validation loss
        if early_stopping(valid_loss):
            print(f"Stopping at epoch {epoch}")
            break  

        if include_test:
            # Run testing
            test_loss, test_iou_score = test_one_epoch(model, test_loader, loss_fn=loss, device=device, discard_allbkgnd = True)
            test_losses.append(test_loss)
            test_ious.append(test_iou_score)
        
        # Update the learning rate
        scheduler.step(valid_loss)
        
        # Save the model if validation loss improves
        if valid_loss < max_score:
            max_score = valid_loss
            torch.save(model, model_filename)
            print('Model saved!')

    toc = time.time()
    elapsed_time = toc - tic
    print(f"\nTraining completed in {elapsed_time // 60:.0f} minutes and {elapsed_time % 60:.2f} seconds.")
    
    return train_losses, train_ious, valid_losses, valid_ious, test_losses, test_ious
 

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last improvement before stopping.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = np.inf
        self.counter = 0

    def __call__(self, current_metric):
        # Check if the current metric is better than the best metric
        if current_metric < self.best_metric - self.min_delta:
            self.best_metric = current_metric
            self.counter = 0  # Reset counter if there is an improvement
            return False  # Do not stop, continue training
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                return True  # Stop training
            return False  # Continue training
            
            


# Initialize AMP GradScaler
scaler = GradScaler('cuda')

# Define training loop function
def train_one_epoch(model, dataloader, optimizer, scaler, loss_fn, device, discard_allbkgnd = True):
    model.train()
    epoch_loss = 0
    total_iou = 0
    num_batches = 0
    iou_metric = JaccardIndex(task='binary', threshold=0.5).to(device)
    nb_blank = 0
    nb_tot_img = 0
    
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)

        # Check which images in the batch have all-background masks
        valid_mask_indices = [i for i in range(masks.size(0)) if masks[i].sum() > 0]
        
        # number of all-background images per batch
        nb_blank += masks.size(0) - len(valid_mask_indices)
        nb_tot_img += masks.size(0)

        # Control all background images
        if discard_allbkgnd == False:
            valid_mask_indices = range(masks.size(0))
            
        # Skip this batch if no valid masks
        if len(valid_mask_indices) == 0:
            print("Skipped batch with all all-background images.")
            continue  

        # Filter images and masks for valid entries
        valid_images = images[valid_mask_indices]
        valid_masks = masks[valid_mask_indices]

        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float32):  # Use float16 precision
            preds = model(valid_images)
            loss = loss_fn(preds, valid_masks)

        # # Mixed Precision Forward Pass
        # with autocast('cuda'):
        #     preds = model(valid_images)
        #     loss = loss_fn(preds, valid_masks)

        # # Forward pass
        # preds = model(valid_images)
        # loss = loss_fn(preds, valid_masks)
        epoch_loss += loss.item()

        
        # Mixed Precision Backward Pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Backward pass and optimization
        # loss.backward()
        # optimizer.step()

        # Calculate IoU for valid images
        total_iou += iou_metric(preds, valid_masks).item() * len(valid_mask_indices)
        num_batches += len(valid_mask_indices)

    # Avoid division by zero if all batches are skipped
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
    avg_iou = total_iou / num_batches if num_batches > 0 else 0
    print(f"Discard 0s? {discard_allbkgnd}\t| 0s_masks/Tot_imgs: {nb_blank}/{nb_tot_img} \t|| Train Loss: {avg_loss:.6f} | Train IoU: {avg_iou:.3f}")
    return avg_loss, avg_iou


def validate_one_epoch(model, dataloader, loss_fn, device, threshold=0.5, discard_allbkgnd=True):
    model.eval()
    epoch_loss = 0
    total_iou = 0.0
    num_valid_images = 0
    iou_metric = JaccardIndex(task='binary', threshold=threshold).to(device)
    nb_blank = 0
    nb_tot_img = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # Check which images in the batch have valid masks
            valid_mask_indices = [i for i in range(masks.size(0)) if masks[i].sum() > 0]

            # number of all-background images per batch
            nb_blank += masks.size(0) - len(valid_mask_indices)
            nb_tot_img += masks.size(0)

            # Control all background images
            if discard_allbkgnd == False:
                valid_mask_indices = range(masks.size(0))
                
            # Skip this batch if no valid masks
            if len(valid_mask_indices) == 0:
                print("Skipped batch with all all-background images.")
                continue  
            
            # Filter images and masks for valid entries
            valid_images = images[valid_mask_indices]
            valid_masks = masks[valid_mask_indices]

            with autocast(device_type='cuda', dtype=torch.float32):  # Use float32 precision
                preds = model(valid_images)
                loss = loss_fn(preds, valid_masks)

            # # Mixed Precision Inference
            # with autocast('cuda'):
            #     preds = model(valid_images)
            #     loss = loss_fn(preds, valid_masks)

            # preds = model(valid_images)
            # loss = loss_fn(preds, valid_masks)
            epoch_loss += loss.item()

            # Threshold predictions for metric computation
            preds = (preds > threshold).int()

            # Calculate IoU for valid images
            total_iou += iou_metric(preds, valid_masks).item() * len(valid_mask_indices)
            num_valid_images += len(valid_mask_indices)

    avg_loss = epoch_loss / num_valid_images if num_valid_images > 0 else 0
    avg_iou = total_iou / num_valid_images if num_valid_images > 0 else 0
    #print(f"0s_masks/Tot_imgs: {nb_blank}/{nb_tot_img} \t|| Valid Loss: {avg_loss:.6f} | Valid IoU: {avg_iou:.3f}")
    print(f"Discard 0s? {discard_allbkgnd}\t| 0s_masks/Tot_imgs: {nb_blank}/{nb_tot_img} \t|| Valid Loss: {avg_loss:.6f} | Valid IoU: {avg_iou:.3f}")
    return avg_loss, avg_iou
    

# Define test loop function
def test_one_epoch(model, dataloader, loss_fn, device, threshold=0.5, discard_allbkgnd = True):
    """
    Test the model for one epoch with a dynamic threshold.
    """
    model.eval()
    epoch_loss = 0
    total_iou = 0.0
    num_valid_images = 0
    #num_batches = 0
    iou_metric = JaccardIndex(task='binary', threshold=threshold).to(device)
    nb_blank = 0
    nb_tot_img = 0

    
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
        
                
            # debugging
            #print(f' deb: {j} masks.size(0) {masks.size(0)}')
            #print(f' deb: {j} images.size(0) {masks.size(0)}')
            
            valid_mask_indices = [i for i in range(masks.size(0)) if masks[i].sum() > 0]
            #print(f' deb: {j} len(valid_mask_indices) {len(valid_mask_indices)}')

            # number of all-background images per batch
            nb_blank += masks.size(0) - len(valid_mask_indices)
            nb_tot_img += masks.size(0)

            # Control all background images
            if discard_allbkgnd == False:
                valid_mask_indices = range(masks.size(0))
            
            # Skip this batch if no valid masks
            if len(valid_mask_indices) == 0:
                print("Skipped batch with all all-background images.")
                continue  
            
            # Filter images and masks for valid entries
            valid_images = images[valid_mask_indices]
            valid_masks = masks[valid_mask_indices]

            # # Mixed Precision Inference
            # with autocast('cuda'):
            #     preds = model(valid_images)
            #     loss = loss_fn(preds, valid_masks)

            with autocast(device_type='cuda', dtype=torch.float32):  # Use float32 precision
                preds = model(valid_images)
                loss = loss_fn(preds, valid_masks)


            # preds = model(valid_images)
            # loss = loss_fn(preds, valid_masks)
            epoch_loss += loss.item()

            # Threshold predictions
            preds = (preds > threshold).int()
            
            # Calculate IoU for valid images
            total_iou += iou_metric(preds, valid_masks).item() * len(valid_mask_indices)
            num_valid_images += len(valid_mask_indices)
            
    avg_loss = epoch_loss / num_valid_images if num_valid_images > 0 else 0
    avg_iou = total_iou / num_valid_images if num_valid_images > 0 else 0
    #print(f"0s_masks/Tot_imgs: {nb_blank}/{nb_tot_img} \t|| Test Loss:  {avg_loss:.6f} | Test IoU: {avg_iou:.3f}")
    print(f"Discard 0s? {discard_allbkgnd}\t| 0s_masks/Tot_imgs: {nb_blank}/{nb_tot_img} \t|| Test Loss: {avg_loss:.6f} | Test IoU: {avg_iou:.3f}")
    return avg_loss, avg_iou
    
    
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0, neg_weight=1.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight  # Weight for positive class
        self.neg_weight = neg_weight  # Weight for negative class

    def forward(self, inputs, targets):
        # print(f"weights: pos {self.pos_weight} neg {self.neg_weight}") # debugging
        # Weighted BCE computation
        loss = -self.pos_weight * targets * torch.log(inputs + 1e-7) - \
               self.neg_weight * (1 - targets) * torch.log(1 - inputs + 1e-7)
        return loss.mean()
        

class BCEFocalNegativeIoULoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=1.5, pos_weight=2.0, neg_weight=1.0):
        """
        Args:
            alpha: Weight for Focal Loss.
            gamma: Modulating factor for Focal Loss.
            pos_weight: Weight for positive class in BCE Loss.
            neg_weight: Weight for negative class in BCE Loss.
        """
        super(BCEFocalNegativeIoULoss, self).__init__()
        self.bce = WeightedBCELoss(pos_weight=pos_weight, neg_weight=neg_weight)
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(self, inputs, targets):
        BCE_loss = -targets * torch.log(inputs + 1e-7) - (1 - targets) * torch.log(1 - inputs + 1e-7)
        #pt = torch.exp(-BCE_loss)  # Probability of the true class
        pt = inputs * targets + (1 - inputs) * (1 - targets)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

    def forward(self, inputs, targets):
        if targets.sum() == 0:  # Skip blank masks
            return torch.tensor(0.0, requires_grad=True).to(inputs.device)

        # Core loss components
        bce_loss = self.bce(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)

        # Compute IoU for positive and negative classes
        #preds = inputs.sigmoid()  # Apply sigmoid activation
        preds = inputs
        #preds = model(inputs)  # No need to apply sigmoid again if already applied in the model
        iou_positive, mean_iou = compute_class_aware_iou(preds, targets)

        # Jaccard Loss for Positive and Negative IoU
        jaccard_loss_positive = 1.0 - iou_positive  # Positive IoU
        jaccard_loss_negative = 1.0 - (2 * mean_iou - iou_positive)  # Derive Negative IoU

        # Weighted Jaccard Loss
        jaccard_loss = 0.85 * jaccard_loss_positive + 0.15 * jaccard_loss_negative

        # Combine all losses
        total_loss = 0.3 * bce_loss + 0.3 * focal_loss + 0.4 * jaccard_loss
        return total_loss


def compute_class_aware_iou(preds, masks, threshold=0.5):
    """
    Compute IoU for positive and negative classes with a dynamic threshold.
    """
    # Apply threshold to predictions
    preds = (preds > threshold).int()
    masks = masks.int()

    preds_np = preds.cpu().numpy().reshape(-1)
    masks_np = masks.cpu().numpy().reshape(-1)

    # Compute IoU for positive and negative classes
    iou_positive = jaccard_score(masks_np, preds_np, pos_label=1)
    iou_negative = jaccard_score(masks_np, preds_np, pos_label=0)

    # Mean IoU
    mean_iou = (iou_positive + iou_negative) / 2
    return iou_positive, mean_iou
