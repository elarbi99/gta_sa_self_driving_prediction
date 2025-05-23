import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b2
from tqdm import tqdm
import numpy as np

def train(model, criterion, optimizer, dataloader, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f'Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

def validate(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    by_action = [0 for i in range(7)]
    by_action_total = [0 for i in range(7)]
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            res = predicted.eq(labels)
            for i in range(7):
                by_action[i] += res[labels == i].sum().item() 
                by_action_total[i] += res[labels == i].size(0)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    by_action = np.array(by_action) / np.array(by_action_total)
    print(f'Val Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    return epoch_acc, by_action

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.0001
    num_classes = 7  # Adjust based on your dataset

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(360),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=360, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_folder = "train"
    test_folder = "test"
    # Datasets and DataLoaders
    train_dataset = datasets.ImageFolder(root=train_folder, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=test_folder, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load EfficientNet and modify the classifier
    model = efficientnet_b2(weights=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    # model.load_state_dict(torch.load("new_model.pt", mmap='cpu', weights_only=True))
    model.to(device)
    # model.eval()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_acc = -torch.inf
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train(model, criterion, optimizer, train_loader, device)
        acc, by_acc = validate(model, criterion, val_loader, device)
        print(by_acc)
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), "new_model.pt")

if __name__ == '__main__':
    main()
