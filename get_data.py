import requests
import json
import os

# https://medium.com/swlh/using-python-to-connect-to-stravas-api-and-analyse-your-activities-dummies-guide-5f49727aac86

# Make Strava auth API call with your
# client_code, client_secret and code
client_id = os.getenv('STRAVA_CLIENT_ID')
client_secret = os.getenv('STRAVA_CLIENT_SECRET')
response = requests.post(
                    url = 'https://www.strava.com/oauth/token',
                    data = {
                            'client_id': client_id,
                            'client_secret': client_secret,
                            'code': 'c37f4816b51beb73806b7186b0c6ddfe9ce58d3b',
                            'grant_type': 'authorization_code'
                            }
)
#Save json response as a variable
strava_tokens = response.json()
# Save tokens to file
with open('strava_tokens.json', 'w') as outfile:
    json.dump(strava_tokens, outfile)
# Open JSON file and print the file contents
# to check it's worked properly
with open('strava_tokens.json') as check:
  data = json.load(check)
print(data)
