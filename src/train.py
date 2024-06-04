import torch
import torch.nn as nn
from torch.optim import Adam
from src.model import CNN
from src.data_utils import load_data, create_data_loaders
from datetime import datetime
import numpy as np

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load and split dataset
data_dir = 'data/raw'
train_subset, val_subset, test_subset = load_data(data_dir, transform)

# Create DataLoaders
batch_size = 64
train_loader, val_loader, test_loader = create_data_loaders(train_subset, val_subset, test_subset, batch_size)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, criterion, and optimizer
model = CNN(len(train_loader.dataset.dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)

    for e in range(epochs):
        t0 = datetime.now()
        train_loss = []
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)

        model.eval()
        val_loss = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss.append(loss.item())
        val_loss = np.mean(val_loss)

        train_losses[e] = train_loss
        val_losses[e] = val_loss
        dt = datetime.now() - t0

        print(f"Epoch: {e+1}/{epochs}, Train_loss: {train_loss:.3f}, Val_loss: {val_loss:.3f}, Duration: {dt}")

    return train_losses, val_losses

# Train the model
train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10)

# Save the trained model
torch.save(model.state_dict(), 'models/plant_disease_model.pt')
