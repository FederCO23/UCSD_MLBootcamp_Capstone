# Step 5: Data Wrangling & Exploration

This repository contains the steps and scripts used to process raw satellite imagery data from the [Brazil Data Cube](https://data.inpe.br/bdc/web/en/home-page-2/), which includes satellite images of the Brazilian territory from various sources. In this project, we focus on imagery from the Sentinel-2 MSI Level-2A program to identify photovoltaic (PV) power plants.

## Procedure Overview

We defined a procedure for data wrangling and exploration, which includes the following steps:

1. **Image Extraction**  
   Extract satellite images containing four bands: red, green, blue, and near-infrared (NIR). These bands are centered on a predefined location of the targeted photovoltaic power plants. The plants vary in size, number at each capture, and surrounding environment.

2. **Capture Window Random Move**  
   We implemented a "Capture Window Random Move" technique, where a moving window slides around the predefined center point to capture multiple images of the plant from different perspectives.

3. **Binary Mask Creation**  
   An automatic procedure generates binary masks to segment the pixels that belong to the solar panels and those of the surroundings.

4. **Manual Mask Refinement**  
   After generating the binary masks, we manually review and refine them to ensure accurate segmentation of the solar cells.

## Jupyter Notebooks

The steps outlined above are detailed in the following Jupyter Notebooks:

- [Data Wrangling](./Data_Wrangling.ipynb) 
  Describes the process for extracting satellite images. 
  
- [Create Mask](./CreateMask.ipynb)

Details the creation of automatic and manual binary masks.
  
- [Capture Windows Random Move](./CaptureWindow_random_move.ipynb)
  Presents the capture window technique and its results.


## Scripts and Libraries

We have also included the libraries and scripts used to perform the batch tasks:

- [Data_tools.py](./Data_tools.py)
  Contains common functions for data wrangling tasks.

- [extractImages.py](./extractImages_script.py)
  A script to batch extract images using the functions from `Data_tools.py`.

- [createMasks_script.py](./createMasks_script.py)
  This script generates initial automatic binary masks for each `.tif` image in the dataset.
  
## Data

This folder contains the generated data set:

--> [link to the Data](https://drive.google.com/drive/folders/1F5sMzaN9w8H9CAqDXCyqiVYR6RTYHNeP?usp=sharing)
