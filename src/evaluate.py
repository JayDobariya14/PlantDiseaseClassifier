import torch
from src.model import CNN
from src.data_utils import load_data, create_data_loaders
from torchvision import transforms

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

# Load the model
model = CNN(len(train_loader.dataset.dataset.classes)).to(device)
model.load_state_dict(torch.load('models/plant_disease_model.pt'))
model.eval()

# Function to calculate accuracy
def accuracy(loader):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            n_correct += (predictions == targets).sum().item()
            n_total += targets.shape[0]
    acc = n_correct / n_total
    return acc

# Evaluate the model
train_acc = accuracy(train_loader)
val_acc = accuracy(val_loader)
test_acc = accuracy(test_loader)

print(f"Train Accuracy: {train_acc * 100:.2f}%")
print(f"Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
