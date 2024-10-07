"""
createMasks_script.py
    This script creates by batch a first automatic mask for every .tif image
"""

from Data_tools import createMask
from pathlib import Path


if __name__ == '__main__':
    # Create a Path object for the current directory
    images_dir = Path('.\\data\\')

    # Define the files prefix
    file_prefix = ('ApioCardoso_4')

    # Get a list of all .tiff files in the current directory
    tif_files = [str(file) for file in images_dir.glob(f'{file_prefix}*.tif')]

    # Create the Mask images
    for tif_file in tif_files:
        print(tif_file + " ---> " + createMask(tif_file, 1800))



