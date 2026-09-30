image_url = request.form["image_url"]

url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}"
params = {
    "api_key": ROBOFLOW_API_KEY,
    "confidence": 40,
    "overlap": 30,
    "image": image_url
}

response = requests.get(url, params=params)
result = response.json()
