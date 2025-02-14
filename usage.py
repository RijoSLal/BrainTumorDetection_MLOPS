import requests

'''
you will get a similar curl like this in fastapi docs when you run the post request

curl -X 'POST' \
  'http://0.0.0.0:8000/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@No19.jpg;type=image/jpeg'

'''




import requests

url = "http://0.0.0.0:8000/"

file_path="brain_tumor_dataset/no/1 no.jpeg"

headers = {
    "accept": "application/json"
}

files = {
    "file": ("img", open(file_path, "rb"), "image/jpeg")
}

response = requests.post(url, headers=headers, files=files)


print(response.json())
