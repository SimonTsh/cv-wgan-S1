# Utilities
import pandas as pd

from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

import requests
from utils.cred import get_access_token, sentinelhub_compliance_hook

from shapely import Polygon, wkt

# Your client credentials
client_id = 'sh-0cfdaa83-7a28-4457-be7f-dbe392bc8c16'
client_secret = 'V8vkeijNyg2QksnrMRYrNOWEDEdSaiXV'

# Create a session
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                          client_secret=client_secret, include_client_id=True)

# All requests using this session will have an access token automatically added
resp = oauth.get("https://sh.dataspace.copernicus.eu/configuration/v1/wms/instances")

oauth.register_compliance_hook("access_token_response", sentinelhub_compliance_hook)


# To find relevant products
aoi_coords_wgs84 = [-46.723480,-23.673141,-46.531906,-23.480882]

start_date = "2024-11-15"
end_date = "2024-11-30"
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

if df.shape[0] > 0 :
    try:
        for i in range(df.shape[0]):
            print(df["Id"][i])
            session = requests.Session()
            
            keycloak_token = get_access_token(username,password)
            session.headers.update({"Authorization": f"Bearer {keycloak_token}"})

            url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({df['Id'][i]})/$value"
            response = session.get(url, allow_redirects=False)
            while response.status_code in (301, 302, 303, 307):
                url = response.headers["Location"]
                response = session.get(url, allow_redirects=False)
            
            try:
                file = session.get(url, verify=True, allow_redirects=True)
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")

            with open(
                f"Code/data/{df['Id'][i]}.zip", #location to save zip from copernicus 
                "wb",
            ) as p:
                print(df["Name"][i])
                p.write(file.content)
    except:
        print("problem with server")

