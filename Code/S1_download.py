# Utilities
import pandas as pd

import pyotp
import requests

from shapely import wkt #, Polygon
from shapely.geometry import Polygon

def get_tokens(username, password, secret_key):
    # Create a TOTP object
    totp = pyotp.TOTP(secret_key)
    current_otp = totp.now() # Generate the current OTP

    # Copernicus Data Space Ecosystem token URL
    url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'client_id': 'cdse-public',
        'username': username,
        'password': password,
        'grant_type': 'password',
        'totp': current_otp
    }
    # Send the request to get the access token
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        access_token = response.json()['access_token']
        refresh_token = response.json()['refresh_token']
    else:
        print(f"Error: {response.status_code}, {response.text}")

    return access_token, refresh_token


# To find relevant products
aoi_coords_wgs84 = [-46.723480,-23.673141,-46.531906,-23.480882]

start_date = "2024-11-15" #"2024-10-15"
end_date = "2024-11-30" #"2024-10-30"
data_collection = "SENTINEL-1"
sensor_mode = "S6" # "IW" # "S6" # "S3" # since SM == S6 & S3
product_type = "SLC__1S" # "GRDH_1S" # "SLC__1S"

min_lon, min_lat, max_lon, max_lat = aoi_coords_wgs84
footprint_polygon = Polygon([
    (min_lon, min_lat),  # Bottom-left
    (max_lon, min_lat),  # Bottom-right
    (max_lon, max_lat),  # Top-right
    (min_lon, max_lat),  # Top-left
    (min_lon, min_lat)   # Close the polygon by repeating the first point
])

# Convert to WKT
footprint_wkt = wkt.dumps(footprint_polygon)
aoi = footprint_wkt #"POLYGON((12.655118166047592 47.44667197521409,21.39065656328509 48.347694733853245,28.334291357162826 41.877123516783655,17.47086198383573 40.35854475076158,12.655118166047592 47.44667197521409))"

json = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=OData.CSC.Intersects(area=geography'SRID=4326;{aoi}') \
    and ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T00:00:00.000Z and Collection/Name eq '{data_collection}' \
        and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{sensor_mode}_{product_type}')").json()
df = pd.DataFrame.from_dict(json["value"])

# To download data product
username = "simon_tsh@hotmail.com"
password = "$nfD=s8,^ZtwL;N"
secret_key = "GJEXO52BMR3FAZSVGI2UWNTXIUZWOSLL"

if df.shape[0] > 0 :
    for i in range(df.shape[0]):
        print(f"Downloading {df['Id'][i]}...")
        url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({df['Id'][i]})/$value"

        # Obtain access_token
        access_token, refresh_token = get_tokens(username, password, secret_key)
        
        # Create a session and update headers
        headers = {"Authorization": f"Bearer {access_token}"}
        session = requests.Session()
        session.headers.update(headers)

        # Perform the GET request
        response = session.get(url, stream=True)
        # response = session.get(url, allow_redirects=False, timeout=timeout)

        # Check if the request was successful
        if response.status_code == 200:
            with open(f"Code/data/{df['Id'][i]}.zip", "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        file.write(chunk)
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
            print(response.text)
            