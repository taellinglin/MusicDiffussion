import os
import torch
import librosa
import soundfile as sf
import numpy as np

from resynthesis import load_model, predict_features, resynthesize_audio_with_prompt
from train import train_model  # Your training script


# Constants
DATA_DIR = "./data"
MODELS_DIR = "./models"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BIT_DEPTH = "PCM_16"
DEFAULT_FORMAT = "wav"


### AUDIO PROCESSING FUNCTIONS ###
def normalize_and_save(mp3_folder, dataset_name, sample_rate, bit_depth, format_choice):
    """Normalize MP3s and save dataset."""
    output_dir = os.path.join(DATA_DIR, dataset_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_combinations = {
        "wav": ["PCM_16", "PCM_24", "PCM_32"],
        "flac": ["PCM_16", "PCM_24", "PCM_32"],
        "mp3": []
    }
    
    if bit_depth not in valid_combinations.get(format_choice, []):
        raise ValueError(f"Invalid combination: '{format_choice}' and '{bit_depth}'.")

    # Normalize and convert MP3s
    for file in os.listdir(mp3_folder):
        if file.endswith(".mp3"):
            audio_path = os.path.join(mp3_folder, file)
            y, sr = librosa.load(audio_path, sr=int(sample_rate))

            output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.{format_choice}")
            sf.write(output_file, y, int(sample_rate), subtype=bit_depth)

    return f"Dataset saved to {output_dir}"


def train_model_ui(model_name, epochs, save_interval, complexity, batch_size, device_str):
    """Train the model and save it in the models folder."""
    model_path = os.path.join(MODELS_DIR, model_name)
    
    dataset_path = os.path.join(DATA_DIR, model_name)
    audio_files = [f for f in os.listdir(dataset_path) if f.endswith(('.wav', '.flac'))]
    device = torch.device(device_str)  # Convert string to torch.device
    # Train model with the loaded dataset
    history = train_model(
        dataset_name=model_name,
        model_name=model_name,
        model_dir=MODELS_DIR,
        epochs=epochs,
        save_interval=save_interval,
        complexity=complexity,
        batch_size=batch_size,
        device=device
    )

    return f"Training complete! Model saved at {model_path}_Master.pth"


### AUDIO GENERATION FUNCTIONS ###
def generate_audio_in_batches(model, full_audio_tensor, focus_window, device, chunk_size=1024):
    """
    Generates audio features in batches to avoid memory overflow and improve efficiency.

    Parameters:
    - model: The PyTorch model
    - full_audio_tensor: The entire input tensor
    - focus_window: Tuple indicating the focus window (start, end)
    - device: The device for inference (CPU or CUDA)
    - chunk_size: Size of each batch

    Returns:
    - Numpy array with concatenated predicted features
    """
    model.eval()

    total_samples = full_audio_tensor.size(0)
    all_predicted_features = []

    # Process audio in batches
    for i in range(0, total_samples, chunk_size):
        batch_tensor = full_audio_tensor[i:i + chunk_size].to(device)

        # 🔥 Use GPU inference
        with torch.no_grad():
            predicted_features_batch = predict_features(model, batch_tensor, focus_window, device)

        # 🔥 Keep results on GPU to avoid repeated cpu-gpu transfers
        all_predicted_features.append(predicted_features_batch)

    # 🔥 Concatenate all batches on GPU first, then move to CPU and convert to NumPy
    if len(all_predicted_features) > 1:
        final_tensor = torch.cat(all_predicted_features, dim=0)
    else:
        final_tensor = all_predicted_features[0]

    # 🚀 Move the final tensor to CPU once and convert to NumPy
    final_array = final_tensor.cpu().numpy()

    return final_array




def generate_audio(model_name, duration, bpm, prompt, device_str='cuda', chunk_size=11136):
    """Generate audio in manageable chunks to avoid memory overflow."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.isfile(model_path):
        return f"❌ Model not found: {model_path}"

    # Set the device based on the string (cuda or cpu)
    device = torch.device(device_str if torch.cuda.is_available() and device_str.startswith("cuda") else "cpu")

    # Load the model
    model = load_model(model_path, device)

    # Generate the audio tensor
    full_audio_tensor = generate_audio_tensor(duration, bpm, prompt, device)

    # Calculate focus window
    focus_window = calculate_focus_window(duration, bpm, model)

    # Process audio in batches
    predicted_features = generate_audio_in_batches(model, full_audio_tensor, focus_window, device, chunk_size)

    # Resynthesize the audio from predicted features
    # Example to ensure both arguments are passed correctly
    # Assuming you have the necessary mel spectrogram features and other parameters
    output_file = resynthesize_audio_with_prompt(
        predicted_features=predicted_features,   # Mel spectrogram features (this comes from your model)
        output_dir=f"output/{model_name}_{duration}_{bpm}_{prompt}",  # Adjust output path as needed
        sample_rate=44100,
        n_iter=32,
        hop_length=5,
        # Ensure this matches the hop length used for STFT
        target_duration=duration  # Duration of the audio (seconds)
    )


    return f"✅ Audio generated: {output_file}"


def calculate_focus_window(duration, bpm, model):
    """Dynamically calculate the focus window."""
    sample_rate = 44100
    total_samples = int(duration * sample_rate)

    window_size = 44100  # Expected feature size (adjust as needed)
    start = 0
    end = min(total_samples, window_size)

    return [start, end]


def generate_audio_tensor(duration, bpm, prompt, device, target_shape=128):  # Set target shape to 256
    """Generate a mel spectrogram tensor with the correct shape."""
    sample_rate = 44100
    num_samples = int(sample_rate * duration)

    # Simulated random waveform input
    audio_tensor = torch.randn(1, num_samples).to(device)  # Move tensor to device
    
    # Convert to numpy for librosa processing
    audio_np = audio_tensor.cpu().numpy().squeeze()

    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_np, sr=sample_rate, n_fft=128, hop_length=32, n_mels=128)  # n_mels=256
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Resize the spectrogram to fit model's shape
    resized_mel = np.resize(log_mel_spec.flatten(), target_shape)

    # Convert back to torch tensor and move to the specified device
    mel_tensor = torch.tensor(resized_mel, dtype=torch.float32).unsqueeze(0).to(device)

    if mel_tensor.shape[1] != target_shape:
        raise ValueError(f"❌ Incorrect tensor shape: {mel_tensor.shape}, expected (1, {target_shape})")

    return mel_tensor
