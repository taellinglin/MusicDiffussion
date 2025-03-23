import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import librosa
import numpy as np
import json
import torch.nn.functional as F


# ✅ Dataset with Flattened Mel Spectrograms
class AudioDataset(Dataset):
    def __init__(self, data_dir, fixed_length=1280, sample_rate=22050, n_mels=128, frames=5):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
        self.fixed_length = fixed_length
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frames = frames
        if not self.files:
            raise ValueError(f"No .wav files found in the directory: {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the audio file with stereo output (2 channels)
        data, _ = librosa.load(self.files[idx], sr=self.sample_rate, mono=False)

        # Ensure the data has 2 channels
        if data.ndim == 1:  # If the audio is mono, create a stereo version by duplicating
            data = np.stack([data, data], axis=0)

        # Calculate mel spectrogram for each channel
        mel_spec_left = librosa.feature.melspectrogram(y=data[0], sr=self.sample_rate, n_mels=self.n_mels)
        mel_spec_right = librosa.feature.melspectrogram(y=data[1], sr=self.sample_rate, n_mels=self.n_mels)

        # Convert to dB scale (log scale)
        log_mel_left = librosa.power_to_db(mel_spec_left, ref=np.max)
        log_mel_right = librosa.power_to_db(mel_spec_right, ref=np.max)

        # Stack the frames and flatten the mel spectrograms for both channels
        mel_spec_stereo = np.concatenate([log_mel_left.flatten(), log_mel_right.flatten()])

        # Ensure the flattened vector is of fixed length (padding/truncating if necessary)
        if mel_spec_stereo.shape[0] < self.fixed_length:
            mel_spec_stereo = np.pad(mel_spec_stereo, (0, self.fixed_length - mel_spec_stereo.shape[0]), mode='constant')
        else:
            mel_spec_stereo = mel_spec_stereo[:self.fixed_length]

        # Return as a torch tensor
        return torch.tensor(mel_spec_stereo, dtype=torch.float32)


# ✅ Model class
class AudioModel(nn.Module):
    def __init__(self, input_dim=None, output_dim=1280):
        super(AudioModel, self).__init__()

        if input_dim is None:
            input_dim = 2 * 128 * 5  # Example, adjust according to your dataset's features
        
        self.fc1 = nn.Linear(input_dim, 128)  # First layer
        self.fc2 = nn.Linear(128, 256)  # Second layer
        self.fc3 = nn.Linear(256, 512)  # Third layer
        self.fc4 = nn.Linear(512, output_dim)  # Fourth layer

        # These layers will accept the Mel spectrogram features
        self.feature_fc3 = nn.Linear(1280, 256)  # Example transformation
        self.feature_fc4 = nn.Linear(1280, 512)  # Example transformation

    def forward(self, x, mel_spec=None):
        # Pass through first two layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Add features to the third layer
        if mel_spec is not None:
            mel_spec_features = self.feature_fc3(mel_spec)  # Transform Mel spectrogram features
            x = torch.cat((x, mel_spec_features), dim=1)  # Concatenate with the output of the second layer
        x = F.relu(self.fc3(x))

        # Add features to the fourth layer
        if mel_spec is not None:
            mel_spec_features = self.feature_fc4(mel_spec)  # Transform Mel spectrogram features
            x = torch.cat((x, mel_spec_features), dim=1)  # Concatenate with the output of the third layer
        x = self.fc4(x)

        return x



# Main training function
def train_model(model_name, dataset_name, model_dir, epochs=100, save_interval=10, complexity=0.5, batch_size=32, device="cuda"):
    data_dir = os.path.join("data/", dataset_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Determine the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define input dimensions, hidden layers, and output dimensions
    input_dim = 128 * 5 * 2  # Mel spectrogram flattened size (128 Mel bins * 5 frames * 2 channels)
    hidden_dim = int(5 * complexity)
    output_dim = 44100 * 2  # Output should match Mel spectrogram dimension

    # Initialize model, optimizer, and loss function
    model = AudioModel(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Create dataset and dataloader
    dataset = AudioDataset(data_dir, fixed_length=input_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Save metadata about the model
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

            # Forward pass: model outputs
            outputs = model(batch)

            # Compute the loss between the outputs and targets (spectrograms)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print the loss for this epoch
        print(f"Epoch {epoch}/{epochs}, Loss: {running_loss:.4f}")

        # Save model checkpoints
        if epoch % save_interval == 0:
            model_checkpoint = os.path.join(model_dir, f"{model_name}_{epoch:03}.pth")
            torch.save(model.state_dict(), model_checkpoint)
            print(f"✅ Saved model checkpoint at {model_checkpoint}")

    # Save final model
    master_model_path = os.path.join(model_dir, f"{model_name}_Master.pth")
    torch.save(model.state_dict(), master_model_path)
    print(f"🎉 Final model saved at {master_model_path}")
