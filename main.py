from fastapi import FastAPI, UploadFile, File

from tensorflow.keras.models import load_model

import io
import cv2
from PIL import Image
import numpy as np 

app = FastAPI()

model = load_model("./models/weights.best.hdf5")

class_names = [
    "Ace", "Akainu", "Brook", "Chopper", "Crocodile", "Franky", 
    "Jinbe", "Kurohige", "Law", "Luffy", "Mihawk", "Nami", 
    "Rayleigh", "Robin", "Sanji", "Shanks", "Ussop", "Zorro"
]

def preprocess_image(image):
    image = cv2.resize(image, (299, 299))
    image = np.expand_dims(image, axis=0)
    image = image.astype("float")/255.
    return image

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    content = await file.read()
    image = np.array(Image.open(io.BytesIO(content)).convert("RGB"))

    image = preprocess_image(image)

    predictions = model.predict(image)
    
    class_name = class_names[np.argmax(predictions[0])]
    class_prob = np.max(predictions[0])

    return {"character": class_name, "confidence": float(class_prob)}