# Step 8: Scaling the ML Prototype

**Objective**: To describe and implement the scaling strategies adopted in our model. Building on the work presented in the previous step, which already incorporates several scaling methods, we aim to detail how these techniques were utilized to manage larger datasets, optimize computational resources, and enhance model performance. This section outlines the key scaling approaches and their practical application in the model's development pipeline.


## Introduction

Scaling the machine learning prototype from a subset of data to a complete dataset presents challenges such as managing memory constraints, leveraging hardware effectively, and balancing trade-offs between computational efficiency and accuracy. This file highlights the scaling techniques described in the linked notebooks, including optimized data handling, GPU utilization, and tailored loss functions, along with the implementation decisions made to address these challenges.

The strategies detailed in the linked notebooks ensure that the model is not only capable of handling a larger dataset but also maximizes resource efficiency and maintains high performance. Each decision is documented to provide insight into the trade-offs involved in scaling.

## Notebooks

- [part 1](../7-Experiment_with_various_Models/PVdetect-modelSelection.ipynb) 
    This notebook demonstrates the initial development of five models, covering:
	
	* Data preparation.
	* Model hyperparameter configuration.
	* Training and evaluation of the models.
	
- [part 2](../7-Experiment_with_various_Models/PVdetect-modelSelection_part2.ipynb) 
    A continuation of the first notebook, focusing on additional experiments, including:
	
	* Data augmentation techniques.
	* Detailed analysis of model performance.
		
- [part 3](../7-Experiment_with_various_Models/PVdetect-modelSelection_part3.ipynb) 
    This notebooks presents the results of Experiencing with Super Resolution images
	
	* Data augmentation techniques.
	* Detailed analysis of model performance.
		
## Applied Techniques

### Efficient Data Handling

- The `torch.utils.data.DataLoader` is configured to optimize the loading of training, validation and test datasets:

```python
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=0)
	valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=0)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=40, shuffle=False, num_workers=0)
```

- Custom transformations are applied to normalize images and masks while maintaining flexibility in dataset preprocessing:

```python
	class ToTensor:
		def __call__(self, image, mask):
			image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
			mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
			return image, mask
```
			
#### Upgrades:
* the data loading is adapted for efficient batch processing and memory optimization.
* Manage the trade-offs of batch sizes, particularly balancing between memory constraints (e.g., small batch sizes for training) and computational efficiency (e.g., larger batch sizes for testing).



### GPU Utilization

Code Implementation: 
* The code dynamically selects the appropriate device (GPU or CPU) for computation, and setting the GPU as the *device* if available: 

```python
	DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

* Mixed precision training is implemented using `torch.cuda.amp.autocast` and `GradScaler` to optimize memory usage and training speed:

```python
	with torch.cuda.amp.autocast(device_type='cuda', dtype=torch.float32):
		preds = model(valid_images)
		loss = loss_fn(preds, valid_masks)

	scaler = torch.cuda.amp.GradScaler()
	scaler.scale(loss).backward()
	scaler.step(optimizer)
	scaler.update()
```

#### Upgrades:
* Mix precision training reduces memory overhead and accelerates computation (GPU usage).
* Fallback mechanism to CPU and GPU for cross-platform usability.



### Handling Class Imbalance

Code Implementation: 
* A custom loss function, `BCEFocalNegativeIoULoss`, addresses class imbalance by combining weighted BCE, Focal Loss, and Jaccard Loss: 

```python
	class BCEFocalNegativeIoULoss(nn.Module):
		def forward(self, inputs, targets):
			bce_loss = WeightedBCELoss(pos_weight=2.0, neg_weight=1.0)(inputs, targets)
			focal_loss = self.focal_loss(inputs, targets)
			jaccard_loss = 0.85 * jaccard_loss_positive + 0.15 * jaccard_loss_negative
			return 0.3 * bce_loss + 0.3 * focal_loss + 0.4 * jaccard_loss
```

#### Upgrades:
* Treat the dataset imbalance(e.g., ~97% negative class) and its implications for model training.
* Adapt the custom loss function to improve rare class detection while balancing precision and recall.



### Model Architecture and Transfer Learning

Code Implementation: 
* The code experiments with multiple segmentation models (U-Net, U-Net++, PSPNet, FPN) and uses pre-trained EfficientNet-b7 encoders:

```python
	model = smp.Unet(
		encoder_name="efficientnet-b7",
		encoder_weights="imagenet",
		in_channels=4,
		classes=1,
		activation="sigmoid"
	)
```

#### Upgrades:
* Experiment with selecting well-performing, lighterweight architectures versus more complex ones.



### Model Architecture and Transfer Learning

Code Implementation: 
* Early stopping is implemented to halt trainig when validation loss stops improving: 

```python
	early_stopping = EarlyStopping(patience=20, min_delta=0.0001)
	if early_stopping(valid_loss):
		print(f"Stopping at epoch {epoch}")
		break
```

* Dynamic learning rate adjustment ensures stable training:

```python
	scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
```

#### Upgrades:
* Early stopping avoids overfitting and saves computational resources and time.
* Dynamic learning rate scheduling for maintaining training stability.


## Conclusion:
The process of scaling and adapting the model provided valuable insights into optimizing our code to maximize the available computing resources. Throughout this journey, we encountered significant challenges, particularly when experimenting with upscaling input samples to improve the level of detail in the data and reduce false positives and negatives.

One major limitation arose when scaling the dataset using upscaling techniques such as Bicubic Interpolation and Super Resolution. Upscaling images by a factor of 2 (Bicubic Interpolation and Super Resolution) and 4 (only Super Resolution) resulted in datasets that were 4 and 16 times larger, respectively. This dramatic increase in data size introduced substantial demands on memory and computing power, revealing the constraints of our GPU hardware.

To address these challenges, several strategies were implemented:
- Early Stopping and a Scheduler were incorporated to achieve better results with fewer epochs, ensuring efficient resource utilization.
- Adjustments to mini-batch sizes proved critical, allowing us to optimize memory usage. However, this came at the cost of reduced precision in some cases.

The results highlighted the impact of these scaling efforts:
1. Using the original dataset (256x256, 4-band GeoTIFF images), we achieved training durations of ~25 minutes for 71 epochs with a mini-batch size of 5.
2. Scaling with Bicubic Interpolation (x2), the dataset increased 4-fold, and training duration extended to 847 minutes over 91 epochs while maintaining the same mini-batch size.
3. Scaling with Super Resolution (x4), the dataset expanded 16-fold, requiring training durations of 2366 minutes (nearly 40 hours) over 70 epochs. This necessitated reducing the mini-batch size to 1 to accommodate memory limitations.

In conclusion, the scaling process emphasized the importance of balancing dataset size, memory usage, and training efficiency. While the larger datasets enabled improved model accuracy and reduced false predictions, they also underscored the hardware constraints that must be managed in future scaling efforts. These experiences serve as a foundation for further optimizations and highlight the necessity of aligning computational resources with the demands of advanced machine learning workflows.
