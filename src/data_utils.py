import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def load_data(data_dir, transform, test_size=0.2, val_size=0.2):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    targets = dataset.targets
    train_idx, test_idx = train_test_split(range(len(targets)), test_size=test_size, stratify=targets)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size / (1 - test_size), stratify=[targets[i] for i in train_idx])
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)
    
    return train_subset, val_subset, test_subset

def create_data_loaders(train_subset, val_subset, test_subset, batch_size):
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader