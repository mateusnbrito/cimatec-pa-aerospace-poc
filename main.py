import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import torch.optim as optim
import matplotlib.pyplot as plt
from os import listdir
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_FILENAME = "aerospace_model.pt"
TRAIN_IMAGES_PATH = "dataset/train"
TEST_IMAGES_PATH = "dataset/test"
PREDICTIONS_IMAGE_PATH = "imagens-de-teste"
BATCH_SIZE = 32
EPOCHS = 1
NUMBER_OF_CLASSES = 6
ENET_OUT_SIZE = 1280
IMAGE_SIZE = 128
LEARNING_RATE = 0.001

image_transformer = None
dataset = None
dataloader = None
model = None
criterion = None
optimizer = None
train_loader = None
classes = None
hasTrainedModel = False

class ImageDataset(Dataset):
  def __init__(self, train_images_dir, transform=None):
    self.data = ImageFolder(train_images_dir, transform=transform)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

  @property
  def classes(self):
    return self.data.classes

class ImageClassifer(nn.Module):
  def __init__(self, num_classes=NUMBER_OF_CLASSES):
    super(ImageClassifer, self).__init__()

    self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
    self.features = nn.Sequential(*list(self.base_model.children())[:-1])

    enet_out_size = ENET_OUT_SIZE

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(enet_out_size, num_classes)
    )

  def forward(self, x):
    x = self.features(x)
    output = self.classifier(x)

    return output

def preprocess_image(image_path, transform):
  image = Image.open(image_path).convert("RGB")

  return image, transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
  model.eval()

  with torch.no_grad():
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)

  return probabilities.cpu().numpy().flatten()

def visualize_predictions(original_image, probabilities, class_names):
  fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

  axarr[0].imshow(original_image)
  axarr[0].axis("off")

  axarr[1].barh(class_names, probabilities)
  axarr[1].set_xlabel("Probabilidade")
  axarr[1].set_title("Predições de Classes")
  axarr[1].set_xlim(0, 1)

  plt.tight_layout()
  plt.show()

def train():
  global model
  global criterion
  global optimizer
  global train_loader
  global test_loader

  train_losses, val_losses = [], []

  for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc="Treinando o modelo..."):
      images, labels = images.to(DEVICE), labels.to(DEVICE)

      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item() * labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    running_loss = 0.0

    with torch.no_grad():
      for images, labels in tqdm(test_loader, desc="Testando o modelo..."):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
      
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * labels.size(0)

    val_loss = running_loss / len(test_loader.dataset)
    val_losses.append(val_loss)

    torch.save(model.state_dict(), MODEL_FILENAME)

    print(f"Epoch { epoch + 1 }/{ EPOCHS } - Perda (Treino): { train_loss }, Perda (Teste): { val_loss }")

def makePredictions():
  global image_transformer
  global model

  for image_name in listdir(PREDICTIONS_IMAGE_PATH):
    image_path = f"{PREDICTIONS_IMAGE_PATH}/{image_name}"

    image, image_tensor = preprocess_image(image_path, image_transformer)
    probabilities = predict(model, image_tensor, DEVICE)

    class_names = dataset.classes

    visualize_predictions(image, probabilities, class_names)

def main():
  global image_transformer
  global dataset
  global dataloader
  global model
  global criterion
  global optimizer
  global train_loader
  global test_loader
  global classes
  global hasTrainedModel

  model_file = Path(MODEL_FILENAME)
  model = ImageClassifer(num_classes=NUMBER_OF_CLASSES)

  if model_file.is_file():
    model.load_state_dict(torch.load(MODEL_FILENAME))

    hasTrainedModel = True

  image_transformer = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
  ])

  dataset = ImageDataset(TRAIN_IMAGES_PATH, image_transformer)
  dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  classes = {v: k for k, v in ImageFolder(TRAIN_IMAGES_PATH).class_to_idx.items()}
  train_dataset = ImageDataset(TRAIN_IMAGES_PATH, transform=image_transformer)
  test_dataset = ImageDataset(TEST_IMAGES_PATH, transform=image_transformer)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

  if not hasTrainedModel:
    train()

  makePredictions()

if __name__=="__main__":
  main()