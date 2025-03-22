import torch
import numpy as np
import librosa
import soundfile as sf
import os
import json
import torch.nn as nn
from datetime import datetime
import random

class DiffusionModel(nn.Module):
    """
    A diffusion model that learns to generate audio from noise, based on mel spectrogram features.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, focus_window_size):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.focus_window_size = focus_window_size
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, focus_window=[11136,51]):
        # Similar to your FocusedModel
        focus_window = [int(focus_window[0]), int(focus_window[1])]
        if len(x.shape) == 2:
            x = x[:, focus_window[0]:focus_window[1]]
        elif len(x.shape) == 3:
            x = x[:, :, focus_window[0]:focus_window[1]]
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def noise_schedule(T, beta_min=0.0001, beta_max=0.02):
    """
    Generate a linear noise schedule.
    T: Number of diffusion timesteps.
    """
    betas = np.linspace(beta_min, beta_max, T)
    alphas = 1 - betas
    alpha_cumprod = np.cumprod(alphas)
    return betas, alphas, alpha_cumprod

def diffuse_audio(mel_spectrogram, T=1000):
    """
    Add noise to the mel spectrogram through the diffusion process.
    mel_spectrogram: The original clean mel spectrogram.
    T: The number of diffusion steps.
    """
    betas, alphas, alpha_cumprod = noise_schedule(T)
    noisy_spectrogram = mel_spectrogram.copy()
    
    # Adding noise over time
    for t in range(T):
        noise = np.random.normal(0, 1, mel_spectrogram.shape)
        noisy_spectrogram = np.sqrt(alpha_cumprod[t]) * noisy_spectrogram + np.sqrt(1 - alpha_cumprod[t]) * noise
    
    return noisy_spectrogram

def denoise(model, noisy_spectrogram, T=1000, focus_window=[11136, 51], device='cpu'):
    """
    Reverse the diffusion process using the model.
    model: The denoising model (DiffusionModel).
    noisy_spectrogram: The noisy input mel spectrogram.
    T: Number of diffusion steps.
    """
    betas, alphas, alpha_cumprod = noise_schedule(T)
    predicted_spectrogram = noisy_spectrogram.copy()

    for t in reversed(range(T)):
        # Convert to tensor and move to device
        tensor_input = torch.tensor(predicted_spectrogram).float().to(device)
        
        # Get the predicted clean version of the spectrogram
        model_output = model(tensor_input, focus_window)
        
        # Update the spectrogram with model output
        noise = model_output.detach().cpu().numpy()
        predicted_spectrogram = (predicted_spectrogram - np.sqrt(1 - alpha_cumprod[t]) * noise) / np.sqrt(alpha_cumprod[t])

    return predicted_spectrogram

def resynthesize_audio(predicted_features, output_dir='./output', sample_rate=44100, n_iter=32, hop_length=512, target_duration=5):
    """
    Resynthesize audio from predicted mel spectrogram features.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure features are in dB scale
    if np.min(predicted_features) < -100:
        mel_db = predicted_features
    else:
        mel_db = librosa.amplitude_to_db(predicted_features)

    n_frames = int(target_duration * sample_rate / hop_length)
    mel_db_resized = mel_db[:n_frames, :]

    # Use librosa to invert the mel spectrogram back to audio (Griffin-Lim)
    audio_waveform = librosa.feature.inverse.mel_to_audio(
        mel_db_resized, 
        sr=sample_rate, 
        hop_length=hop_length, 
        n_iter=n_iter
    )

    output_file = os.path.join(output_dir, 'resynthesized_audio.wav')
    sf.write(output_file, audio_waveform, sample_rate)

    return output_file

def predict_features(model, audio_tensor, focus_window, device):
    """
    Predict features from the model with input validation.
    """
    expected_shape = model.fc1.in_features
    if audio_tensor.shape[1] != expected_shape:
        raise ValueError(f"Invalid input shape: {audio_tensor.shape[1]}, expected {expected_shape}")

    model.eval()
    with torch.no_grad():
        predicted_features = model(audio_tensor, focus_window).cpu().numpy()

    return predicted_features
