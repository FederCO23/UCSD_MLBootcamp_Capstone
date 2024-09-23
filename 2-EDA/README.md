# Step 2: Collect Your Data

This repository contains the objectives for the **Step 2 – Collect Your Data** Capstone Submission.

## Data Sources

The two main data sources used for this project are:

### 1. [Brazil Data Cube](https://data.inpe.br/bdc/web/en/home-page-2/) 
<img src="./sup_images/logo-bdc.png" align="right" width="64" />

Brazil Data Cube is a research, development, and technological innovation project of the National Institute for Space Research (INPE), Brazil. It produces data sets from large volumes of medium-resolution remote sensing images for the entire national territory. The project also develops a computational platform to process and analyze these data sets using artificial intelligence, machine learning, and image time series analysis.

### 2. [Google Earth Engine](https://earthengine.google.com/)
<img src="./sup_images/logo-gee.png" align="right" width="64" />

Google Earth Engine combines a multi-petabyte catalog of satellite imagery and geospatial datasets with planetary-scale analysis capabilities. It is widely used by scientists, researchers, and developers to detect changes, map trends, and quantify differences on the Earth's surface. While it is available for commercial use, it remains free for academic and research purposes.

## Data Access and Exploratory Data Analysis

For accessing the data from both platforms, I have prepared two Jupyter notebooks, which also include an initial Exploratory Data Analysis (EDA) over the available images:

- [Notebook 1 - Brazil Data Cube Data Access](./BDC_EDA.ipynb)
- [Notebook 2 - Google Earth Engine Data Access](./GEE_EDA.ipynb)

## Image Data

I have created two subfolders containing images from both sources. These images are in the GEO TIFF format and require specific software, such as QGIS, to be visualized.

- [Brazil Data Cube Images](https://drive.google.com/drive/folders/1lg493XvS7nrm1Jowp3T_FNFsMjbXZz5e?usp=drive_link)
- [Google Earth Engine Images](./GEE_images/)

To illustrate the image extraction, here’s a sample visualized using QGIS software:

<img src="./sup_images/2024-09-15 10_40_29-_compare_BDC-GEE_extraction — QGIS.png" align="center" width="512" />


## Requirements

- QGIS software for visualizing GEO TIFF images.
- Access to Google Earth Engine and Brazil Data Cube platforms for further data exploration.

---

Thank you for reviewing my project. If you have any questions, feel free to reach out via GitHub Issues or Discussions.
