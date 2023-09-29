import requests

# Define the URL of your DRF API endpoint
api_url = 'http://localhost:8000/analyze_sentiment/'  # Replace with your actual API URL

# Define the input data as a dictionary
data = {"comment": "It's awesome!!"}

# Send a POST request with the JSON data
response = requests.post(api_url, json=data)

# Print the response from the API
print("Response Status Code:", response.status_code)
print("Response JSON:", response.json())
