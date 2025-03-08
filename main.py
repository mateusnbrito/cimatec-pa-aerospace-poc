from os import listdir
from PIL import Image
import numpy as np

import random

dataset_path = "dataset/"

classes = [
  {
    "id": 0,
    "name": "Avião de Caça",
    "path": "aviao_caca"
  },
  {
    "id": 1,
    "name": "Avião Comercial",
    "path": "aviao_comercial"
  },
  {
    "id": 2,
    "name": "Balão",
    "path": "balao"
  },
  {
    "id": 3,
    "name": "Dirigível",
    "path": "dirigivel"
  },
  {
    "id": 4,
    "name": "Helicóptero",
    "path": "helicoptero"
  },
  {
    "id": 5,
    "name": "Ônibus Espacial",
    "path": "onibus_espacial"
  }
]

dataset = []

def addImagesToClasses():
  for a_class in classes:
    images_paths = f"{dataset_path}{a_class['path']}"

    for image_name in listdir(images_paths):
      image_path = f"{images_paths}/{image_name}"

      image = Image.open(image_path)

      image = image.resize((360,360))

      # image.save(f"images/{random.randrange(0, 3000, 1)}.jpg")

      image_array = np.asarray(image)

      dataset.append([image_array.tolist(), a_class["id"]])

addImagesToClasses()