import requests
import json
import os

# https://medium.com/swlh/using-python-to-connect-to-stravas-api-and-analyse-your-activities-dummies-guide-5f49727aac86

# Copied code goes here:
copied_code = ''
# Make Strava auth API call with your
# client_code, client_secret and code
client_id = os.getenv('STRAVA_CLIENT_ID')
client_secret = os.getenv('STRAVA_CLIENT_SECRET')
response = requests.post(
                    url = 'https://www.strava.com/oauth/token',
                    data = {
                            'client_id': client_id,
                            'client_secret': client_secret,
                            'code': copied_code,
                            'grant_type': 'authorization_code'
                            }
)
#Save json response as a variable then save to file
strava_tokens = response.json()
with open('./data/strava_tokens.json', 'w') as outfile:
    json.dump(strava_tokens, outfile)
