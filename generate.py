import torch
import librosa
import numpy as np
import soundfile as sf

class AudioModel(torch.nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=32):
        super(AudioModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def extract_features(audio_path, duration=5, sample_rate=44100):
    """
    Extracts features from an audio file.
    Args:
        audio_path: Path to audio file.
        duration: Seconds to extract features from.
        sample_rate: Sample rate for loading audio.
    """
    y, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
    
    # Extract MFCCs as dummy features (replace with your feature extraction)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
    
    # Flatten and normalize
    features = np.mean(mfccs, axis=1)
    return torch.tensor(features, dtype=torch.float32)

def generate_features(audio_path, model_path, duration=5):
    """
    Predict audio features using trained model.
    Args:
        audio_path: Path to audio file.
        model_path: Path to trained model.
        duration: Seconds of audio to process.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = AudioModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Extract features
    features = extract_features(audio_path, duration).to(device)

    # Predict features
    with torch.no_grad():
        predicted = model(features.unsqueeze(0))

    return predicted.cpu().numpy()
