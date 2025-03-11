import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN_IMAGES_PATH = "dataset/train"
TEST_IMAGES_PATH = "dataset/test"
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

    for images, labels in tqdm(train_loader, desc="Training loop"):
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
      for images, labels in tqdm(test_loader, desc="Test loop"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
      
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * labels.size(0)

    val_loss = running_loss / len(test_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch { epoch + 1 }/{ EPOCHS } - Perda (Treino): { train_loss }, Perda (Teste): { val_loss }")

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

  image_transformer = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
  ])

  dataset = ImageDataset(TRAIN_IMAGES_PATH, image_transformer)
  dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
  model = ImageClassifer(num_classes=NUMBER_OF_CLASSES)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  classes = {v: k for k, v in ImageFolder(TRAIN_IMAGES_PATH).class_to_idx.items()}

  train_dataset = ImageDataset(TRAIN_IMAGES_PATH, transform=image_transformer)
  test_dataset = ImageDataset(TEST_IMAGES_PATH, transform=image_transformer)

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

  train()

if __name__=="__main__":
  main()