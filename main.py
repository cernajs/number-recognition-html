from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage
import numpy as np
import io
import base64
import pickle
import lzma

app = FastAPI()
pickle_in = open("classifier.pkl", "rb")
model = pickle.load(pickle_in)

class ImageData(BaseModel):
    image: str

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/recognize")
async def recognize(image_data: ImageData):
    try:
        image = PILImage.open(io.BytesIO(base64.b64decode(image_data.image.split(',')[1])))
        image = image.resize((28, 28))

        image_array = np.zeros((28, 28))

        for x, y in [(x, y) for x in range(image.size[0]) for y in range(image.size[1])]:
            item = image.getpixel((x, y))
            sub = 0
            if item[3] != 0:
                sub = 255

            image_array[x][y] = sub
        
        normalized_image_array = image_array

        # mean = normalized_image_array.mean().astype(np.float32)
        # std = normalized_image_array.std().astype(np.float32)
        # normalized_image_array = (normalized_image_array - mean)/(std)

        for x, y in [(x, y) for x in range(normalized_image_array.shape[0]) for y in range(normalized_image_array.shape[1])]:
            if normalized_image_array[x][y] < 200:
                normalized_image_array[x][y] = 0
            else:
                normalized_image_array[x][y] = 255

        #temp_image_array = (normalized_image_array * 255).astype(np.uint8)
        #temp_img = PILImage.fromarray(temp_image_array.T)
        #temp_img.save("static/image.png")

        print(normalized_image_array.T)


        prediction = model.predict(normalized_image_array.T.reshape(1, 784))

        print(prediction.tolist())

        #prediction = np.exp(prediction - np.max(prediction))
        #prediction /= prediction.sum()

        #return .2f precision
        #prediction = np.round(prediction, 10)

        return {'prediction': prediction.tolist()}

    except Exception as e:
        return {'error': str(e)}
