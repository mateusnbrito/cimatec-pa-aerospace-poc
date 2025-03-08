import numpy as np
from os import listdir
from PIL import Image

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

image_width = 360
image_height = 360

xDataset = []
yDataset = []

def addImagesToClasses():
  for a_class in classes:
    images_paths = f"{dataset_path}{a_class['path']}"

    for image_name in listdir(images_paths)[:5]:
      image_path = f"{images_paths}/{image_name}"

      image = Image.open(image_path)

      image = image.resize((360,360))

      image_array = np.asarray(image)

      image_array = image_array.reshape(-1, 9)

      xDataset.append(image_array)
      yDataset.append(a_class["id"])

def trainModel():
  print(xDataset)
  print(yDataset)

addImagesToClasses()
trainModel()