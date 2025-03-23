import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from transformers import BertTokenizer, BertModel  # For prompt tokenization and embedding
import os
import torchaudio
import json

class FocusedModel(nn.Module):
    """
    A modified model that uses a window of focus to select relevant parts of the input
    based on tokens in the prompt (could represent temporal or feature-wise focus).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, focus_window_size, token_embedding_dim):
        super(FocusedModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.focus_window_size = focus_window_size
        self.token_embedding_dim = token_embedding_dim
        
        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Layer for mel spectrograms (right)
        self.fc2 = nn.Linear(input_dim, hidden_dim)  # Layer for mel spectrograms (left)
        self.fc3 = nn.Linear(token_embedding_dim, hidden_dim)  # Layer for token embeddings (third layer)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)  # Layer for token-weighted features (fourth layer)
        self.fc5 = nn.Linear(hidden_dim, output_dim)  # Final output layer
        
        # ReLU activation function
        self.relu = nn.ReLU()
    
    def forward(self, mel_right, mel_left, prompt_embedding):
        # Step 1: Process the mel spectrograms (left and right)
        mel_right = self.fc1(mel_right)
        mel_left = self.fc2(mel_left)
        
        # Step 2: Token embedding processing
        token_features = self.fc3(prompt_embedding)
        
        # Step 3: Integrate the token features (weights)
        combined_features = mel_right + mel_left + token_features
        
        # Step 4: Process the combined features through the next layers
        x = self.fc4(combined_features)
        x = self.relu(x)
        x = self.fc5(x)
        
        return x

    def resynthesize(self, features, sample_rate=44100):
        """
        Resynthesize audio waveform from predicted features.

        Parameters:
        - features (Tensor): Predicted features tensor
        - sample_rate (int): Sampling rate for audio

        Returns:
        - waveform (Tensor): Generated audio waveform
        """
        # Ensure features are PyTorch tensors
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float().to(features.device)

        # Apply waveform generation using sine wave (this is an example; modify for actual model)
        waveform = torch.sin(2 * torch.pi * features).sum(dim=1).unsqueeze(0)
        
        # Resample to target sample rate (if needed, though it is the same here)
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sample_rate)(waveform)

        return waveform

    def save_to_file(self, waveform, output_path, sample_rate=44100):
        """
        Save the generated waveform to a file.

        Parameters:
        - waveform (Tensor): The generated waveform
        - output_path (str): Path to save the audio file
        - sample_rate (int): Sampling rate for the audio file
        """
        # Convert waveform from Tensor to numpy array
        waveform = waveform.squeeze().cpu().numpy()

        # Use soundfile to write the audio to the file
        sf.write(output_path, waveform, sample_rate)
def predict_features_with_prompt(model, mel_right_tensor, mel_left_tensor, prompt_text, device):
    """
    Predict features from the model with input validation and prompt text integration.

    Parameters:
    - model: The model to use for prediction.
    - mel_right_tensor: The mel spectrogram (right) tensor.
    - mel_left_tensor: The mel spectrogram (left) tensor.
    - prompt_text: The prompt text (such as genre) for token weighting.
    - device: The device to run the model on.

    Returns:
    - The predicted features as a numpy array.
    """
    # Tokenize the prompt text (e.g., genre) using a pre-trained BERT model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True)
    prompt_embedding = BertModel.from_pretrained('bert-base-uncased')(inputs['input_ids']).last_hidden_state.mean(dim=1)
    
    # Ensure tensors are on the correct device
    mel_right_tensor = mel_right_tensor.to(device)
    mel_left_tensor = mel_left_tensor.to(device)
    prompt_embedding = prompt_embedding.to(device)
    
    # Ensure the tensor shapes are correct (batch_size, features)
    model.eval()
    with torch.no_grad():
        predicted_features = model(mel_right_tensor, mel_left_tensor, prompt_embedding).cpu().numpy()

    return predicted_features

def resynthesize_audio_with_prompt(predicted_features, output_dir='./output', sample_rate=44100, n_iter=32, hop_length=5, target_duration=5):
    """
    Resynthesize audio from predicted mel spectrogram features with prompt-based weighting.
    
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
    try:
        audio_waveform = librosa.feature.inverse.mel_to_audio(
            mel_db_resized, 
            sr=sample_rate, 
            hop_length=hop_length, 
            n_iter=n_iter
        )
    except MemoryError:
        print("Error: Memory allocation failure during inversion. Try reducing input size or using a smaller hop length.")
        return None

    # Save the waveform as a WAV file
    output_file = os.path.join(output_dir, 'resynthesized_audio_with_prompt.wav')
    sf.write(output_file, audio_waveform, sample_rate)

    print(f"✅ Audio saved at: {output_file}")
    return output_file

def load_model(model_path, device=None):
    """
    Load the PyTorch model from the specified path, handling state_dict refitting.

    Parameters:
    - model_path: Path to the model file.
    - device: Device to load the model on (CPU or CUDA). Defaults to 'cuda' if available.

    Returns:
    - PyTorch model loaded onto the specified device with refitted layers.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"🔧 Loading model from {model_path} on {device}...")

    # Load metadata
    metadata_path = model_path.replace('_Master.pth', '_metadata.json')
    
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata file not found: {metadata_path}")
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Extract model parameters
    input_dim = metadata.get('input_dim', 11136)
    hidden_dim = metadata.get('hidden_dim', 512)
    output_dim = metadata.get('output_dim', 1280)
    token_embedding_dim = metadata.get('token_embedding_dim', 256)
    focus_window_size = metadata.get('focus_window_size', 5)


    model = FocusedModel(
        input_dim, 
        hidden_dim, 
        output_dim, 
        focus_window_size, 
        token_embedding_dim
    )

    # Load the checkpoint
    try:
        model_data = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"❌ Error loading model data: {e}")
        return None

    if not isinstance(model_data, dict):
        print("❌ Invalid model format.")
        return None

    checkpoint = model_data

    print("📦 Refitting model state_dict...")

    # Create a new state_dict to refit mismatched dimensions
    new_state_dict = model.state_dict()

    for key, param in checkpoint.items():
        if key in new_state_dict:
            target_shape = new_state_dict[key].shape

            if param.shape != target_shape:
                print(f"🔧 Resizing layer: {key} from {param.shape} to {target_shape}")

                # Handle 1D biases
                if param.dim() == 1 and len(target_shape) == 1:
                    if param.shape[0] < target_shape[0]:
                        # Pad with zeros
                        padding_size = target_shape[0] - param.shape[0]
                        resized_param = torch.nn.functional.pad(
                            param, (0, padding_size)
                        )
                    else:
                        # Truncate
                        resized_param = param[:target_shape[0]]

                # Handle 2D weights
                elif param.dim() == 2 and len(target_shape) == 2:
                    try:
                        resized_param = torch.nn.functional.adaptive_avg_pool2d(
                            param.unsqueeze(0), target_shape
                        ).squeeze(0)
                    except Exception as e:
                        print(f"⚠️ Fallback to random init due to error: {e}")
                        resized_param = torch.rand(target_shape)

                # Handle unexpected dimensions
                else:
                    print(f"⚠️ Fallback to random init for {key} due to shape mismatch.")
                    resized_param = torch.rand(target_shape)

                # Assign resized parameter
                new_state_dict[key] = resized_param.to(device)
            else:
                new_state_dict[key] = param.to(device)

        else:
            print(f"⚠️ Skipping unexpected key: {key}")

    # Load the refitted state_dict into the model
    model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model.eval()

    print("✅ Model loaded with refitted layers successfully!")
    return model

def generate_audio_in_batches(model, full_audio_tensor, focus_window, device, chunk_size):
    """
    Generate audio in batches by processing chunks of the input audio.

    Parameters:
    - model: The pre-trained model used for audio generation.
    - full_audio_tensor: The full tensor representing the audio.
    - focus_window: A tuple (start, end) indicating which portion of the audio to focus on.
    - device: The device to run the model on (CPU or CUDA).
    - chunk_size: Size of the audio chunks to process at once.

    Returns:
    - predicted_features: The features predicted by the model.
    """
    predicted_features_batch = []
    
    for start in range(0, full_audio_tensor.size(1), chunk_size):
        end = min(start + chunk_size, full_audio_tensor.size(1))
        audio_chunk_tensor = full_audio_tensor[:, start:end]
        
        # Pass the focus_window to the predict_features function
        predicted_features = predict_features(model, audio_chunk_tensor, focus_window=focus_window, device=device)
        predicted_features_batch.append(predicted_features)

    return torch.cat(predicted_features_batch, dim=1)



import torch
import soundfile as sf
import numpy as np
import os
import librosa

def resynthesize_audio(model, predicted_features, output_path, duration_sec, sample_rate=44100):
    """
    Resynthesize audio by iterating over the duration and generating waveform chunks for each second,
    calling the model's forward method every iteration to update the tensor.
    
    Parameters:
    - model: The model used for audio generation
    - predicted_features: Features predicted by the model
    - output_path: Path where the audio should be saved
    - duration_sec: Total duration of the audio in seconds
    - sample_rate: Sampling rate for the output file
    """
    # Initialize an empty list to accumulate the waveform chunks
    total_waveform = []

    # Calculate the total number of samples based on the desired duration (in seconds)
    num_samples = int(duration_sec * sample_rate)  # Total number of samples needed

    # Loop through each chunk of audio (1 second at a time)
    for i in range(num_samples):
        print(predicted_features)
        # Extract mel_left and prompt_embedding from predicted_features (update as needed)
        mel_left = predicted_features  # Replace with actual extraction
        prompt_embedding = predicted_features  # Replace with actual extraction

        # Pass mel_left and prompt_embedding to the model's forward function
        chunk = model(mel_left, prompt_embedding)  # Forward pass with required inputs

        # Optionally, reshape or adjust the chunk if necessary
        # Ensure the chunk is properly shaped (e.g., [1, num_samples])
        chunk = chunk.unsqueeze(0)  # Ensure it's in the correct shape for concatenation
        
        # Append the chunk to the total waveform list
        total_waveform.append(chunk)

        # Optionally, print progress
        if i % 100 == 0:
            print(f"Generating chunk {i}/{num_samples}...")

    # Convert the accumulated waveform list into a single tensor
    total_waveform = torch.cat(total_waveform, dim=1)

    # Check the waveform shape and some sample values
    print(f"Final waveform shape: {total_waveform.shape}")
    print(f"First few waveform samples: {total_waveform[0, :10]}")

    # Ensure the waveform is within the valid audio range
    total_waveform = total_waveform.clamp(-1.0, 1.0)

    # Replace invalid characters in output file name
    valid_output_path = output_path.replace("::", "_").replace(" ", "_")  # Replace invalid chars

    # Ensure the directory exists
    output_dir = os.path.dirname(valid_output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the generated waveform to the file (ensure it's numpy for saving)
    sf.write(valid_output_path, total_waveform.cpu().numpy().flatten(), sample_rate)

    print(f"Audio saved to {valid_output_path}")


    
    
def predict_features(model, audio_chunk_tensor, focus_window, device):
    """
    Predicts features using the model with proper input reshaping.

    Parameters:
    - model: The PyTorch model
    - audio_chunk_tensor: The audio chunk tensor
    - focus_window: Tuple indicating the focus window (start, end)
    - device: The device for inference (CPU or CUDA)

    Returns:
    - The predicted features tensor
    """
    model.eval()

    # Move tensors to device
    audio_chunk_tensor = audio_chunk_tensor.to(device)

    # Dummy tensors for mel_left and prompt_embedding
    batch_size = audio_chunk_tensor.size(0)

    # 🔥 Match dimensions to refitted model layers
    mel_size = model.fc2.in_features  # Match input dimension of fc2
    prompt_dim = model.token_embedding_dim  # Match token embedding dimension

    # Ensure proper tensor shapes
    mel_left = torch.zeros((batch_size, mel_size), device=device)
    prompt_embedding = torch.zeros((batch_size, prompt_dim), device=device)

    # 🔥 Reshape the input tensor to match model's input layer
    input_dim = model.input_dim
    if audio_chunk_tensor.shape[1] != input_dim:
        print(f"⚠️ Reshaping input from {audio_chunk_tensor.shape} to ({batch_size}, {input_dim})")
        audio_chunk_tensor = torch.nn.functional.interpolate(
            audio_chunk_tensor.unsqueeze(1),  # Add a channel dimension
            size=(input_dim,),
            mode='linear',
            align_corners=False
        ).squeeze(1)  # Remove the extra channel dimension

    # Perform inference with reshaped tensor
    with torch.no_grad():
        features = model(audio_chunk_tensor, mel_left, prompt_embedding)

    return features

# ✅ Add a resynthesize method to the model
    