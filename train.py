import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import librosa
import numpy as np
import json


# ✅ Dataset with Flattened Mel Spectrograms
class AudioDataset(Dataset):
    def __init__(self, data_dir, fixed_length=44100, sample_rate=22050, n_mels=128):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
        self.fixed_length = fixed_length
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        if not self.files:
            raise ValueError(f"No .wav files found in the directory: {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data, _ = librosa.load(self.files[idx], sr=self.sample_rate)

        # Pad or truncate audio to fixed length
        if len(data) < self.fixed_length:
            data = np.pad(data, (0, self.fixed_length - len(data)), mode='constant')
        else:
            data = data[:self.fixed_length]

        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=data, sr=self.sample_rate, n_mels=self.n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Flatten mel spectrogram to 11136 dimensions
        mel_spec_flat = log_mel_spec.flatten()

        return torch.tensor(mel_spec_flat, dtype=torch.float32)


# ✅ Model predicting spectrograms (11136 dimensions)
class AudioModel(nn.Module):
    def __init__(self, input_dim=11136, hidden_dim=512, output_dim=11136):  
        super(AudioModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Match mel spec output
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  
        return self.layers(x)


# ✅ Training Function
def train_model(model_name, dataset_name, model_dir, epochs=100, save_interval=10, complexity=0.5, batch_size=32):
    data_dir = os.path.join("data/", dataset_name)
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 11136  # Mel spectrogram size
    hidden_dim = int(512 * complexity)
    output_dim = 11136  # Match mel spectrogram dimensions

    # Model and DataLoader
    model = AudioModel(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    dataset = AudioDataset(data_dir, fixed_length=44100)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Save model metadata
    metadata = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "complexity": complexity,
        "sample_rate": 44100,
        "batch_size": batch_size
    }

    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    print(f"📄 Saved model metadata at {metadata_path}")

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()

            # Model outputs 11136 dimensions
            outputs = model(batch)

            # Ensure target and output have same dimension
            loss = criterion(outputs, batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch}/{epochs}, Loss: {running_loss:.4f}")

        if epoch % save_interval == 0:
            model_checkpoint = os.path.join(model_dir, f"{model_name}_{epoch:03}.pth")
            torch.save(model.state_dict(), model_checkpoint)
            print(f"✅ Saved model checkpoint at {model_checkpoint}")

    # Save final master model
    master_model_path = os.path.join(model_dir, f"{model_name}_Master.pth")
    torch.save(model.state_dict(), master_model_path)
    print(f"🎉 Final model saved at {master_model_path}")
