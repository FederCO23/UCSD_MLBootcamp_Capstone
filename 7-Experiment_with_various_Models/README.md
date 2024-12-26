# Step 7: Experiment With Various Models

This repository contains the code, models, and supporting files developed to experiment with various approaches for modeling our dataset. The goal is to establish benchmarks and identify the most effective models for detecting photovoltaic solar panels in satellite imagery using semantic segmentation techniques.

### Tested Models and Results

The performance of the tested models is summarized in the following table:

<img src="./sup_images/results.png" align="center" width="2048" />

## Repository Contents

### 1. Main Jupyter Notebooks

- [part 1](./PVdetect-modelSelection.ipynb) 
    This notebook demonstrates the initial development of five models, covering:
		- Data preparation.
		- Model hyperparameter configuration.
		- Training and evaluation of the models.
	
- [part 2](./PVdetect-modelSelection_part2.ipynb) 
    A continuation of the first notebook, focusing on additional experiments, including:
		- Data augmentation techniques.
		- Detailed analysis of model performance.

### 2. Model Tuning Log

- [history logs](./modelTuning_logs.pdf)
    A document describing the iterative model tuning process. It includes:
		- Insights into hyperparameter adjustments (learning rates, batch sizes, loss functions).
		- Performance metrics across various configurations.
		- Strategies for addressing class imbalance and overfitting.
    This document serves as a reference for understanding the trade-offs and impacts of different parameter settings.

### 3. Supporting Library

- [tools lib](./S7_tools.py)
    A Python module containing utility functions and custom classes used throughout the project. Key functionalities include:
		- Training and validation loops.
		- Visualization of loss and IoU metrics over epochs.
		- Custom loss functions (weighted BCE and Focal loss).
		- Confusion matrix computation and visualization.
			
			
			
