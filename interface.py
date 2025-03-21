import gradio as gr
import os
import torch
import librosa
import soundfile as sf
import numpy as np
import shutil

from dataset import process_mp3_folder  # Your MP3 processing function
from train import train_model  # Your training script
from generate import generate_features  # Your feature generation script

# Constants
DEFAULT_MP3_DIR = "./mp3"
DATA_DIR = "./data"
MODELS_DIR = "./models"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BIT_DEPTH = "16bit"
DEFAULT_FORMAT = "wav"

### DATA TAB FUNCTIONS ###
def normalize_and_save(mp3_folder, dataset_name, sample_rate, bit_depth, format_choice):
    """ Normalize MP3s and save dataset """
    output_dir = os.path.join(DATA_DIR, dataset_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Normalize and convert MP3s
    for file in os.listdir(mp3_folder):
        if file.endswith(".mp3"):
            audio_path = os.path.join(mp3_folder, file)
            y, sr = librosa.load(audio_path, sr=sample_rate)
            
            # Convert to specified format
            output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.{format_choice}")
            sf.write(output_file, y, sample_rate, subtype=bit_depth)

    return f"Dataset saved to {output_dir}"

### TRAIN TAB FUNCTIONS ###
def train_model_ui(model_name, epochs, save_interval, complexity):
    """ Train the model and save it in models folder """
    model_path = os.path.join(MODELS_DIR, model_name)
    
    # Train model
    history = train_model(
        data_dir=DATA_DIR,
        model_path=model_path,
        epochs=epochs,
        save_interval=save_interval,
        complexity=complexity
    )
    
    return f"Training complete! Model saved at {model_path}_Master.pth"

### GENERATE TAB FUNCTIONS ###
def predict_features(audio_file, model_name, duration):
    """ Predict audio features using the model """
    model_path = os.path.join(MODELS_DIR, model_name)
    
    # Convert if needed
    audio_ext = audio_file.name.split('.')[-1]
    if audio_ext not in ['mp3', 'wav', 'flac']:
        return "Unsupported format. Please upload MP3, WAV, or FLAC."
    
    # Extract features
    features = generate_features(audio_file.name, model_path, duration)
    
    return f"Predicted features: {features}"


### GRADIO INTERFACE ###
with gr.Blocks() as app:
    gr.Markdown("# 🎵 Audio Processing Interface")

    with gr.Tabs():
        # DATA TAB
        with gr.Tab("Data"):
            gr.Markdown("## 🎧 Normalize and Save Dataset")
            
            mp3_folder = gr.Textbox(label="MP3 Folder", value=DEFAULT_MP3_DIR)
            dataset_name = gr.Textbox(label="Dataset Name", placeholder="Enter dataset name")
            
            sample_rate = gr.Slider(8000, 48000, value=DEFAULT_SAMPLE_RATE, label="Sample Rate (Hz)")
            bit_depth = gr.Dropdown(["16bit", "24bit", "32bit"], value=DEFAULT_BIT_DEPTH, label="Bit Depth")
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

            model_name = gr.Textbox(label="Model Name", placeholder="Enter model name")
            epochs = gr.Slider(1, 1000, value=100, label="Epochs")
            save_interval = gr.Slider(1, 100, value=10, label="Save every X epochs")
            complexity = gr.Slider(0.1, 1.0, value=0.5, label="Training Complexity (Speed vs Accuracy)")

            train_button = gr.Button("Train")
            output_train = gr.Textbox(label="Training Status")

            train_button.click(
                fn=train_model_ui,
                inputs=[model_name, epochs, save_interval, complexity],
                outputs=output_train
            )

        # GENERATE TAB
        with gr.Tab("Generate"):
            gr.Markdown("## 🎤 Generate Features")
            
            audio_file = gr.File(label="Upload Audio (MP3/WAV/FLAC)")
            model_select = gr.Dropdown(
                choices=[m for m in os.listdir(MODELS_DIR) if m.endswith(".pth")],
                label="Select Model"
            )
            
            duration = gr.Slider(1, 30, value=5, label="Prediction Duration (seconds)")
            
            generate_button = gr.Button("Generate")
            output_generate = gr.Textbox(label="Predicted Features")
            
            generate_button.click(
                fn=predict_features,
                inputs=[audio_file, model_select, duration],
                outputs=output_generate
            )

app.launch()
