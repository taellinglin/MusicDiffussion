import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Dummy Dataset Loader (replace with your actual dataset loader)
class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return torch.tensor(data, dtype=torch.float32)

# Simple NN Model (replace with your actual architecture)
class AudioModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=32):
        super(AudioModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(data_dir, model_path, epochs=100, save_interval=10, complexity=0.5):
    """
    Train audio model.
    Args:
        data_dir: Path to dataset folder.
        model_path: Path to save the model.
        epochs: Number of training epochs.
        save_interval: Save model every X epochs.
        complexity: Affects hidden layer size (0.1 = fast, 1.0 = accurate).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure complexity
    input_dim = 128
    hidden_dim = int(64 * complexity)
    output_dim = 32

    # Model and DataLoader
    model = AudioModel(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    dataset = AudioDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch)
            loss = criterion(outputs, batch)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch {epoch}/{epochs}, Loss: {running_loss:.4f}")

        # Save model every X epochs
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), f"{model_path}_{epoch:03}.pth")
            print(f"✅ Saved model checkpoint at {model_path}_{epoch:03}.pth")

    # Save the final master model
    torch.save(model.state_dict(), f"{model_path}_Master.pth")
    print(f"🎉 Final model saved at {model_path}_Master.pth")
