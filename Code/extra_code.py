import requests

from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from utils.cred import get_access_token, sentinelhub_compliance_hook

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

session = requests.Session()
timeout = (5, 30)  # 5 seconds for connection, 30 seconds for reading
while response.status_code in (301, 302, 303, 307):
    url = response.headers["Location"]
    response = session.get(url, allow_redirects=False, timeout=timeout)

try:
    file = session.get(url, verify=True, allow_redirects=True, timeout=timeout)
except requests.exceptions.Timeout:
    print(f"Request timed out for {df['Id'][i]}")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

with open(
    f"Code/data/{df['Id'][i]}.zip", # location to save zip from copernicus 
    "wb",
) as p:
    print(df['Name'][i])
    p.write(file.content)
