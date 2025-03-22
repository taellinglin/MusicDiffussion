import torch
import numpy as np
import librosa
import soundfile as sf
import os
import json
import torch.nn as nn
from datetime import datetime  # Import datetime module

class FocusedModel(nn.Module):
    """
    A modified model that uses a window of focus to select relevant parts of the input
    based on tokens in the prompt (could represent temporal or feature-wise focus).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, focus_window_size):
        super(FocusedModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.focus_window_size = focus_window_size
        
        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, focus_window=[11136,51]):
        # Validate that focus_window contains valid integer values
        try:
            # Ensure the values in focus_window are integers
            focus_window = [int(focus_window[0]), int(focus_window[1])]
        except ValueError as e:
            # If invalid values are present, raise a detailed error
            raise ValueError(f"Invalid value in focus_window: {focus_window}. Must be integers.") from e

        print(f"x shape before slicing: {x.shape}")
        
        # Handle slicing for 2D and 3D tensors
        if len(x.shape) == 2:  # If it's 2D
            x = x[:, focus_window[0]:focus_window[1]]
        elif len(x.shape) == 3:  # If it's 3D
            x = x[:, :, focus_window[0]:focus_window[1]]
        else:
            raise ValueError(f"Input tensor has unsupported shape: {x.shape}")
        
        print(f"x shape after slicing: {x.shape}")
        
        # Flatten the input to match the model architecture
        x = x.view(x.size(0), -1)  # Flatten to a 2D tensor (batch_size, features)
        
        # Feed the data through the layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

def load_model(model_path, focus_window_size=512, device=None):
    """
    Load a PyTorch model from the specified path, handling both full models and state_dicts,
    incorporating a focus window for attention-based selection.

    Parameters:
    - model_path: Path to the model file.
    - focus_window_size: Size of the window of focus on the input.
    - device: Device to load the model on (CPU or CUDA). Defaults to 'cuda' if available.

    Returns:
    - PyTorch model loaded onto the specified device.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🔧 Loading model from {model_path} on {device}...")

    # Load the state_dict and metadata
    try:
        with open(model_path.replace('_Master.pth', '_Master_metadata.json'), 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"❌ Metadata file for {model_path} not found!")
        return None
    except json.JSONDecodeError:
        print(f"❌ Failed to decode JSON in metadata file for {model_path}")
        return None

    input_dim = metadata.get('input_dim', None)
    hidden_dim = metadata.get('hidden_dim', None)
    output_dim = metadata.get('output_dim', None)
    
    if not all([input_dim, hidden_dim, output_dim]):
        print(f"❌ Incomplete metadata found for model: {model_path}")
        return None

    # Load the model data (either full model or state_dict)
    try:
        model_data = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"❌ Error loading model data from {model_path}: {e}")
        return None

    # Build the model with a focus window parameter
    model = FocusedModel(input_dim, hidden_dim, output_dim, focus_window_size)

    if isinstance(model_data, dict):  # If it's a state_dict
        print("📦 Model saved as state_dict. Loading state_dict into model...")

        # Create a mapping of the state_dict keys to match the model's layers
        new_state_dict = {}
        for k, v in model_data.items():
            # Check if the key starts with 'layers.' and map it to the model's layer names
            if k.startswith("layers.0."):
                new_state_dict["fc1." + k.split("layers.0.")[1]] = v
            elif k.startswith("layers.2."):
                new_state_dict["fc2." + k.split("layers.2.")[1]] = v
            else:
                # If there's an unexpected key, we can either skip it or print a warning
                print(f"⚠️ Unexpected key in state_dict: {k}")

        # Now load the remapped state_dict into the model
        try:
            model.load_state_dict(new_state_dict)
            print("✅ State_dict loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading state_dict: {e}")
            return None

    else:  # Full model
        model = model_data

    # Move the model to the specified device
    model.to(device)
    model.eval()

    # Additional logging for debugging
    print(f"Model Architecture: {model}")
    print("✅ Model loaded successfully!")
    
    return model

def save_model(model, model_name, output_dir='./models'):
    """
    Save the PyTorch model with a datetime timestamp.
    
    Parameters:
    - model: The model to save.
    - model_name: The base name of the model.
    - output_dir: Directory to save the model.
    
    Returns:
    - Path to the saved model file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current date and time for the timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Create the filename with the model name and timestamp
    filename = f"{model_name}_{timestamp}.pth"
    model_path = os.path.join(output_dir, filename)
    
    # Save the model state_dict
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved at: {model_path}")
    
    return model_path

def predict_features(model, audio_tensor, focus_window, device):
    """
    Predict features from the model with input validation.

    Parameters:
    - model: The model to use for prediction.
    - audio_tensor: The input audio tensor.
    - focus_window: The window size for inference.
    - device: The device to run the model on.

    Returns:
    - The predicted features as a numpy array.
    """
    # Ensure the tensor has the correct shape
    expected_shape = model.fc1.in_features  # Get model input shape dynamically
    if audio_tensor.shape[1] != expected_shape:
        raise ValueError(
            f"Invalid input shape: {audio_tensor.shape[1]}, expected {expected_shape}"
        )

    # Run inference
    model.eval()
    with torch.no_grad():
        # Apply focus window and get features
        predicted_features = model(audio_tensor, focus_window).cpu().numpy()  # Ensure it's on CPU for further processing

    return predicted_features

def resynthesize_audio(predicted_features, output_dir='./output', sample_rate=44100, n_iter=32, hop_length=512, target_duration=5):
    """
    Resynthesize audio from predicted mel spectrogram features.
    
    Parameters:
    - predicted_features: Numpy array of predicted mel spectrogram features (shape: [n_frames, n_mels])
    - output_dir: Directory to save the generated audio file.
    - sample_rate: Sample rate of the output audio.
    - n_iter: Number of iterations for Griffin-Lim algorithm.
    - hop_length: Hop length used for inversion (should match the one used for STFT).
    - target_duration: Desired duration of the output audio in seconds.
    
    Returns:
    - Path to the resynthesized audio file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Invert mel spectrogram to waveform using Griffin-Lim
    print("🔄 Inverting mel spectrogram to waveform...")

    # Ensure features are in dB scale
    if np.min(predicted_features) < -100:  # Check if already in dB scale
        mel_db = predicted_features
    else:
        mel_db = librosa.amplitude_to_db(predicted_features)

    # Calculate the number of frames needed for the target duration
    n_frames = int(target_duration * sample_rate / hop_length)
    
    # Truncate or pad the mel spectrogram to match the target duration
    mel_db_resized = mel_db[:n_frames, :]  # Slice or pad to match the number of frames

    # Use librosa to invert the mel spectrogram back to audio (using Griffin-Lim)
    audio_waveform = librosa.feature.inverse.mel_to_audio(
        mel_db_resized, 
        sr=sample_rate, 
        hop_length=hop_length, 
        n_iter=n_iter
    )

    # Save the waveform as a WAV file
    output_file = os.path.join(output_dir, 'resynthesized_audio.wav')
    sf.write(output_file, audio_waveform, sample_rate)

    print(f"✅ Audio saved at: {output_file}")
    return output_file
