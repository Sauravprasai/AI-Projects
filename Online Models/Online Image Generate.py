from openai import OpenAI
from apikey import apikey
import os
from PIL import Image
import requests
from io import BytesIO

os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey

client = OpenAI()
prompt = "Lion flying in a cloud"

response = client.images.generate( model="dall-e-3",
                                   prompt=prompt,
                                   size="1024x1024",
                                   n=1 )

image_url = response.data[0].url
image = requests.get(image_url)
image = Image.open(BytesIO(image.content))

image.save("AI_Generated_Image.png")
image.show()