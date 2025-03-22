import gradio as gr
import os
import torch
import librosa
import soundfile as sf
import numpy as np
import shutil
import argparse

from dataset import process_mp3_folder  # Your MP3 processing function
from train import train_model  # Your training script
from generate import generate_features  # Your feature generation script
from resynthesis import load_model, predict_features, resynthesize_audio
from util import AudioGenerator
# Constants
DEFAULT_MP3_DIR = "./mp3"
DATA_DIR = "./data"
MODELS_DIR = "./models"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BIT_DEPTH = "PCM_16"
DEFAULT_FORMAT = "wav"


### DATA TAB FUNCTIONS ###
def normalize_and_save(mp3_folder, dataset_name, sample_rate, bit_depth, format_choice):
    """ Normalize MP3s and save dataset """
    output_dir = os.path.join(DATA_DIR, dataset_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Validate bit depth and format choice compatibility
    valid_combinations = {
        "wav": ["PCM_16", "PCM_24", "PCM_32"],
        "flac": ["PCM_16", "PCM_24", "PCM_32"],
        "mp3": []  # MP3 typically does not use bit depth
    }
    
    if bit_depth not in valid_combinations.get(format_choice, []):
        raise ValueError(f"Invalid combination of format '{format_choice}' and bit depth '{bit_depth}'.")

    # Normalize and convert MP3s
    for file in os.listdir(mp3_folder):
        if file.endswith(".mp3"):
            audio_path = os.path.join(mp3_folder, file)
            y, sr = librosa.load(audio_path, sr=int(sample_rate))

            # Convert to specified format
            output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.{format_choice}")
            sf.write(output_file, y, int(sample_rate), subtype=bit_depth)

    return f"Dataset saved to {output_dir}"

### TRAIN TAB FUNCTIONS ###
def train_model_ui(model_name, epochs, save_interval, complexity, batch_size):
    """ Train the model and save it in models folder """
    model_path = os.path.join(MODELS_DIR, model_name)
    
    # Load dataset
    dataset_path = os.path.join(DATA_DIR, model_name)
    audio_files = [f for f in os.listdir(dataset_path) if f.endswith(('.wav', '.flac'))]
    
    # Train model with the loaded dataset
    history = train_model(
        dataset_name=model_name,  # Use model_name instead of dataset_name

        model_name=model_name,
        model_dir="models\\",
        epochs=epochs,
        save_interval=save_interval,
        complexity=complexity,
        batch_size=batch_size
    )
    
    return f"Training complete! Model saved at {model_path}_Master.pth"


### GRADIO INTERFACE ###
def parse_args():
    parser = argparse.ArgumentParser(description="Launch Gradio Interface")
    parser.add_argument("--listen", type=str, default="127.0.0.1", help="IP address to listen on")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    return parser.parse_args()

args = parse_args()

with gr.Blocks() as app:
    gr.Markdown("# 🎵 Audio Processing Interface")

    with gr.Tabs():
        # DATA TAB
        with gr.Tab("Data"):
            gr.Markdown("## 🎧 Normalize and Save Dataset")
            
            mp3_folder = gr.Textbox(label="MP3 Folder", value=DEFAULT_MP3_DIR)
            dataset_name = gr.Textbox(label="Dataset Name", placeholder="Enter dataset name")
            
            sample_rate = gr.Dropdown(["44100", "48000", "96000", "192000"], value="44100", label="Sample Rate")

            bit_depth = gr.Dropdown(["PCM_16", "PCM_24", "PCM_32"], value=DEFAULT_BIT_DEPTH, label="Bit Depth")
            format_choice = gr.Dropdown(["wav", "mp3", "flac"], value=DEFAULT_FORMAT, label="Output Format")

            normalize_button = gr.Button("Normalize and Save")
            output_data = gr.Textbox(label="Output")
            
            normalize_button.click(
                fn=normalize_and_save,
                inputs=[mp3_folder, dataset_name, sample_rate, bit_depth, format_choice],
                outputs=output_data
            )

        # TRAIN TAB
        with gr.Tab("Train"):
            gr.Markdown("## 🛠️ Train Model")

            model_name = gr.Dropdown(choices=os.listdir(DATA_DIR), label="Select Dataset")

            model_name_input = gr.Textbox(label="Model Name", placeholder="Enter model name")
            epochs = gr.Slider(1, 1000, value=100, label="Epochs")
            save_interval = gr.Slider(1, 100, value=10, label="Save every X epochs")
            complexity = gr.Slider(0.1, 1.0, value=0.5, label="Training Complexity (Speed vs Accuracy)")

            batch_size = gr.Slider(1, 128, value=32, label="Batch Size")
            train_button = gr.Button("Train")
            output_train = gr.Textbox(label="Training Status")

            train_button.click(
                fn=train_model_ui,
                inputs=[model_name, epochs, save_interval, complexity, batch_size],
                outputs=output_train
            )

        

        def generate_audio_in_batches(model, full_audio_tensor, focus_window, device, chunk_size=44100):
            """
            Generate audio features in batches with correct tensor shape.

            Parameters:
            - model: The model used for generation.
            - full_audio_tensor: Full audio tensor to process in chunks.
            - focus_window: Focus window size.
            - device: Device to run on (CUDA or CPU).
            - chunk_size: Size of each chunk (default: 44100).

            Returns:
            - Combined predicted features.
            """
            num_samples = full_audio_tensor.shape[1]
            num_batches = (num_samples + chunk_size - 1) // chunk_size
            
            all_predicted_features = []

            for batch_idx in range(num_batches):
                start_idx = batch_idx * chunk_size
                end_idx = min((batch_idx + 1) * chunk_size, num_samples)

                # Slice the audio tensor for the current chunk
                audio_chunk = full_audio_tensor[:, start_idx:end_idx]

                # ✅ Convert CUDA tensor to CPU before using NumPy
                audio_chunk_cpu = audio_chunk.cpu().numpy()

                # Pad the chunk if it is smaller than the chunk size
                if audio_chunk_cpu.shape[1] < chunk_size:
                    audio_chunk_cpu = np.pad(audio_chunk_cpu, ((0, 0), (0, chunk_size - audio_chunk_cpu.shape[1])), mode='constant')

                # ✅ Convert back to PyTorch tensor and move to CUDA
                audio_chunk_tensor = torch.tensor(audio_chunk_cpu, dtype=torch.float32).to(device)

                # Run inference
                predicted_features_batch = predict_features(model, audio_chunk_tensor, focus_window=focus_window, device=device)

                all_predicted_features.append(predicted_features_batch)

            return np.concatenate(all_predicted_features, axis=1)


        def generate_audio(model_name, duration, bpm, prompt, device='cuda', chunk_size=11136):
            """
            Generate audio in manageable chunks to avoid memory overflow.
            
            Parameters:
            - model_name: The name of the model.
            - duration: Duration of audio to generate.
            - bpm: Beats per minute for rhythm.
            - prompt: Prompt to guide generation.
            - device: The device (default is 'cuda').
            - chunk_size: Size of chunks to process at once (default is 44100).
            
            Returns:
            - The generated audio file.
            """
            # Validate model path
            model_path = os.path.join(MODELS_DIR, model_name)
            if not os.path.isfile(model_path):
                return f"❌ Model not found: {model_path}"

            # Load the model
            model = load_model(model_path, device)

            # Generate the audio tensor (use a real method for this)
            full_audio_tensor = generate_audio_tensor(duration, bpm, prompt, device)
            
            # Calculate focus window based on audio generation needs
            focus_window = calculate_focus_window(duration, bpm, model)

            # Process audio in batches
            predicted_features = generate_audio_in_batches(model, full_audio_tensor, focus_window, device, chunk_size)

            # Resynthesize the audio from predicted features
            output_file = resynthesize_audio(predicted_features)

            return f"✅ Audio generated: {output_file}"

        def calculate_focus_window(duration, bpm, model):
            """
            Dynamically calculate the focus window based on duration, bpm, and model requirements.
            """
            sample_rate = 44100  # Assuming model expects 44100 Hz sample rate
            total_samples = int(duration * sample_rate)

            # Adjust focus window size based on model's input
            window_size = 44100  # Expected feature size (adjust based on your model's needs)
            start = 0
            end = min(total_samples, window_size)

            return [start, end]



        def generate_audio_tensor(duration, bpm, prompt, device, target_shape=11136):
            """
            Generate a mel spectrogram tensor with the correct shape for the model.

            Parameters:
            - duration (float): Duration of the audio in seconds.
            - bpm (float): Beats per minute (for rhythm adjustments).
            - prompt (str): Text prompt for audio generation.
            - device (str): 'cpu' or 'cuda'.
            - target_shape (int): Expected model input shape.

            Returns:
            - mel_tensor (torch.Tensor): Resized mel spectrogram tensor with shape (1, 11136).
            """
            sample_rate = 44100
            num_samples = int(sample_rate * duration)

            # Generate random audio waveform (simulated input)
            audio_tensor = torch.randn(1, num_samples).to(device)
            
            # Convert to numpy for librosa processing
            audio_np = audio_tensor.cpu().numpy().squeeze()

            # Generate mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio_np, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=128)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            # Resize the spectrogram to fit the model's expected shape
            resized_mel = np.resize(log_mel_spec.flatten(), target_shape)

            # Convert back to torch tensor
            mel_tensor = torch.tensor(resized_mel, dtype=torch.float32).unsqueeze(0).to(device)

            # Validate the shape
            if mel_tensor.shape[1] != target_shape:
                raise ValueError(f"❌ Incorrect tensor shape: {mel_tensor.shape}, expected (1, {target_shape})")

            return mel_tensor



        # Gradio interface
        with gr.Tab("Generate"):
            gr.Markdown("## 🎤 Generate Audio")

            model_select = gr.Dropdown(
                choices=[m for m in os.listdir(MODELS_DIR) if m.endswith(".pth")],
                label="Select Model"
            )

            duration = gr.Slider(1, 30, value=5, label="Duration (seconds)")
            bpm = gr.Slider(60, 180, value=120, label="BPM")
            prompt = gr.Textbox(label="Prompt", placeholder="Enter a prompt for audio generation")
            # File input for .wav, .mp3, and .flac formats
            file_input = gr.File(label="Upload Audio File", file_types=[".wav", ".mp3", ".flac"])
            generate_button = gr.Button("Generate")
            output_generate = gr.Textbox(label="Generated Audio Path")

            # Trigger audio generation pipeline
            generate_button.click(
                fn=generate_audio,
                inputs=[model_select, duration, bpm, prompt],
                outputs=output_generate
            )

# Launch the Gradio app with the provided listen and port arguments
app.launch(server_name=args.listen, server_port=args.port, share=True)
