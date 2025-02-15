"""
extractImages_script.py
    extract images by batch using the Data_tools lib
"""

from Data_tools import captureWindow
import pystac_client

if __name__ == '__main__':
    # Define the global variables for the captures
    service = pystac_client.Client.open('https://data.inpe.br/bdc/stac/v1/')  # using Brazil Data Cube as data source
    center_point = (-8.041142, -35.012587)  # PV plant coordinates
    place_name = "SãoLorenço"
    output_path = ".\\data\\"
    nb_images = 20  # number of captures
    offset_distance_km = 1.5  # Define the distance to extend from the center_point for the main bounding box
    date_interval = '2023-11-01/2023-12-31'  # date interval for cloud treatment

    captureWindow(service,
              center_point,
              place_name,
              output_path,
              nb_images,
              offset_distance_km,
              date_interval)