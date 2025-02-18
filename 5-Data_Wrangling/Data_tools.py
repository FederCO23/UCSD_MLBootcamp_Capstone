"""
Data_tools.py
    Common functions for Data Wrangler
"""


import pystac_client
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform
from rasterio.windows import from_bounds
from rasterio.mask import mask
from geopy.distance import geodesic
from matplotlib import pyplot as plt
from rasterio.windows import Window, transform as window_transform
seed_value = 23
np.random.seed(seed_value)


# Compute median bands to mitigate the cloud distortion
def compute_median_band(band_data_list):
    data_stack = ma.stack(band_data_list, axis=0)
    median_band = ma.median(data_stack, axis=0)
    return median_band

# We have implemented a `read` method that allows to retrieve part of an image according to a rectangle specified
# in `EPSG:4326` (latitude and longitude).
def read_multiple_items(items, band_name, bbox, masked=True, crs=None):

    source_crs = CRS.from_string('EPSG:4326')
    if crs:
        source_crs = CRS.from_string(crs)

    data_list = []
    transforms = []
    crs_list = []

    for item in items:
        uri = item.assets[band_name].href

        # Expects the bounding box has 4 values
        w, s, e, n = bbox

        with rasterio.open(uri) as dataset:
            # Transform the bounding box to the dataset's CRS
            xs, ys = transform(source_crs, dataset.crs, [w, e], [s, n])

            # Create a window from the transformed bounds
            window = from_bounds(xs[0], ys[0], xs[1], ys[1], dataset.transform)

            # Read the data within the window
            data = dataset.read(1, window=window, masked=masked)

            # Get the transform for the windowed data
            window_transform = dataset.window_transform(window)

            # Get the CRS of the dataset
            data_crs = dataset.crs

            data_list.append(data)
            transforms.append(window_transform)
            crs_list.append(data_crs)

    return data_list, transforms, crs_list



def captureWindow(service, main_centroid, place_name, output_path, nb_images, offset_distance_km, date_interval):

    # Calculate the main bounding box where the CaptureWindow will move and take the captures
    main_bb_north = geodesic(kilometers=offset_distance_km).destination(main_centroid, 0).latitude
    main_bb_south = geodesic(kilometers=offset_distance_km).destination(main_centroid, 180).latitude
    main_bb_east = geodesic(kilometers=offset_distance_km).destination(main_centroid, 90).longitude
    main_bb_west = geodesic(kilometers=offset_distance_km).destination(main_centroid, 270).longitude
    main_bbox = (main_bb_west, main_bb_south, main_bb_east, main_bb_north)

    print(f"Main Bounding box limits:\nSouth: {main_bb_south} \tWest: {main_bb_west}\nNorth: {main_bb_north} \tEast: {main_bb_east}")
    print(80 * "-")

    # -----------------------------
    # Initialize the CaptureWindow
    # ----------------------------

    for i in range(nb_images):

        sub_centroid = (np.random.uniform(main_bb_south, main_bb_north), np.random.uniform(main_bb_west, main_bb_east))
        distance_km = 1.35

        sub_bb_north = geodesic(kilometers=distance_km).destination(sub_centroid, 0).latitude
        sub_bb_south = geodesic(kilometers=distance_km).destination(sub_centroid, 180).latitude
        sub_bb_east = geodesic(kilometers=distance_km).destination(sub_centroid, 90).longitude
        sub_bb_west = geodesic(kilometers=distance_km).destination(sub_centroid, 270).longitude
        sub_bbox = (sub_bb_west, sub_bb_south, sub_bb_east, sub_bb_north)


        # -----------------------
        # Treat Cloud Distortion.
        # -----------------------
        # To minimize the clouds distortion, let's define a search time interval from july to
        # August (in general, less cloudy season).

        item_search = service.search(bbox=sub_bbox,
                                     datetime=date_interval,
                                     collections=['S2-16D-2'])

        print(f'Number of images in the collection: {item_search.matched()}')

        # Convert item_search.items() to a list to reuse
        items_list = list(item_search.items())

        # Consider four bands: red, green, blue and NIR.
        # Observe that the read function consider a collection of images
        red_data_list, red_transforms, red_crs_list = read_multiple_items(items_list, 'B04', sub_bbox)
        green_data_list, green_transforms, green_crs_list = read_multiple_items(items_list, 'B03', sub_bbox)
        blue_data_list, blue_transforms, blue_crs_list = read_multiple_items(items_list, 'B02', sub_bbox)
        nir_data_list, nir_transforms, nir_crs_list = read_multiple_items(items_list, 'B08', sub_bbox)

        # Compute median band values to absorb cloud distortions
        median_red = compute_median_band(red_data_list)
        median_green = compute_median_band(green_data_list)
        median_blue = compute_median_band(blue_data_list)
        median_nir = compute_median_band(nir_data_list)

        # Prepare median bands for writing
        # force float32 data type for later treatment with TensorFlow
        nodata_value = -9999.0
        median_red_filled = median_red.filled(nodata_value).astype('float32')
        median_green_filled = median_green.filled(nodata_value).astype('float32')
        median_blue_filled = median_blue.filled(nodata_value).astype('float32')
        median_nir_filled = median_nir.filled(nodata_value).astype('float32')

        # Setting the image size as 256x256 pixels
        # Extract the first 256 rows and columns for each band
        median_red_subset = median_red_filled[0:256, 0:256]
        median_green_subset = median_green_filled[0:256, 0:256]
        median_blue_subset = median_blue_filled[0:256, 0:256]
        median_nir_subset = median_nir_filled[0:256, 0:256]

        # Create a Window object for the subset
        window = Window(0, 0, 256, 256)

        # Compute the new affine transform for the subset
        reference_transform = red_transforms[0]
        new_transform = window_transform(window, reference_transform)

        # Stack the median subsets
        stacked_median_array = np.stack([
            median_red_subset,
            median_green_subset,
            median_blue_subset,
            median_nir_subset
        ])

        # Update the output height and width
        output_height, output_width = median_red_subset.shape

        output_filename = output_path + place_name + "_" + str(i+1) + ".tif"
        reference_crs = red_crs_list[0]

        # Write the output file
        with rasterio.open(
                output_filename,
                'w',
                driver='GTiff',
                height=output_height,
                width=output_width,
                count=4,
                dtype='float32',
                crs=reference_crs,
                transform=new_transform,
                nodata=nodata_value
        ) as dst:
            dst.write(stacked_median_array)
            dst.set_band_description(1, 'Red')
            dst.set_band_description(2, 'Green')
            dst.set_band_description(3, 'Blue')
            dst.set_band_description(4, 'NIR')

        # log
        print(f'image file: ' + output_filename+ " created")
        print(80 * "-")


def createMask(file_name, threshold=1500):

    with rasterio.open(file_name) as dataset:
        # Read the NIR band (band 4)
        nir_band = dataset.read(4)
        # Get metadata if needed
        nir_meta = dataset.meta
        nodata_value = dataset.nodata

    # Ensure nodata_value is a float
    nodata_value = float(nodata_value) if nodata_value is not None else -9999.0

    # Create a boolean mask for valid data
    valid_mask = nir_band != nodata_value

    # Initialize the binary mask with the nodata_value as float32
    binary_mask = np.full(nir_band.shape, nodata_value, dtype='float32')

    # Apply the threshold only to valid data, using float values
    binary_mask[valid_mask] = np.where(nir_band[valid_mask] < threshold, 1.0, 0.0)

    # Save the binary mask to a GeoTIFF file
    binary_meta = nir_meta.copy()
    binary_meta.update({
        'dtype': 'float32',
        'count': 1,
        'nodata': nodata_value
    })

    output_file = file_name[:-4] + "_automask" + ".tif"
    with rasterio.open(output_file, 'w', **binary_meta) as dst:
        dst.write(binary_mask, 1)

    return output_file




