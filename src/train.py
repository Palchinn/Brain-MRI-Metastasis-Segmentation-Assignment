import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.dataset import BrainMRIDataset
from models.nested_unet import NestedUNet
from models.attention_unet import AttentionUNet
from utils.metrics import dice_score

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_dice = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_score(outputs, masks).item()

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_dice += dice_score(outputs, masks).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    batch_size = 8
    num_epochs = 100
    learning_rate = 0.001

    # Data loaders
    train_dataset = BrainMRIDataset('data/processed', split='train', transform=transforms.ToTensor())
    val_dataset = BrainMRIDataset('data/processed', split='test', transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = NestedUNet(num_classes=1, input_channels=1).to(device)
    # Uncomment the following line to train Attention U-Net instead
    # model = AttentionUNet(num_classes=1, input_channels=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)