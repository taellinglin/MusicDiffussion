import os
import gradio as gr
import argparse
from audio_proc import normalize_and_save, train_model_ui, generate_audio
from params import load_model_parameters
import torch
# Constants
DEFAULT_MP3_DIR = "./mp3"
DATA_DIR = "./data"
MODELS_DIR = "./models"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BIT_DEPTH = "PCM_16"
DEFAULT_FORMAT = "wav"
def refresh_model_dropdown():
    # List the files with a .pth extension in the models directory
    model_files = [m for m in os.listdir(MODELS_DIR) if m.endswith(".pth")]
    return gr.Dropdown.update(choices=model_files)
def refresh_dropdown():
    # List the files in the directory and update the dropdown choices
    dataset_files = os.listdir(DATA_DIR)
    return gr.Dropdown.update(choices=dataset_files)
# Argument parser for launching
def parse_args():
    parser = argparse.ArgumentParser(description="Launch Gradio Interface")
    parser.add_argument("--listen", type=str, default="127.0.0.1", help="IP address to listen on")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    return parser.parse_args()

args = parse_args()


# Helper function for loading parameters with visualization
def load_params_with_viz(model_path):
    params, plots, layer_params = load_model_parameters(model_path)
    return params, plots, layer_params

def get_device(device_str):
    if device_str == "CPU":
        return "cpu"
    else:
        # Return the string representing the GPU device
        return device_str  # Example: "cuda:0"



# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("# 🎵 Audio Processing Interface")

    with gr.Tabs():
        # DATA TAB
        with gr.Tab("Data"):
            gr.Markdown("## 🎧 Normalize and Save Dataset")

            mp3_folder = gr.Textbox(label="MP3 Folder", value="./mp3")
            dataset_name = gr.Textbox(label="Dataset Name")
            sample_rate = gr.Dropdown(["44100", "48000", "96000"], value="44100", label="Sample Rate")
            bit_depth = gr.Dropdown(["PCM_16", "PCM_24", "PCM_32"], value="PCM_16", label="Bit Depth")
            format_choice = gr.Dropdown(["wav", "flac"], value="wav", label="Format")

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
# Get the list of available GPU devices
            gpu_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.device_count() > 0 else ["CPU"]

            # Create the dropdown with available GPUs
            device_id = gr.Dropdown(choices=gpu_devices, value="cuda:0" if gpu_devices else "CPU", label="Select GPU")
            model_name = gr.Dropdown(choices=os.listdir(DATA_DIR), label="Select Dataset")
                # Button to trigger the refresh
            refresh_button = gr.Button("Refresh Dropdown")
            
            # Set the refresh button to update the dropdown when clicked
            refresh_button.click(refresh_dropdown)
                # Button to trigger the refresh
            epochs = gr.Slider(1, 1000, value=100, label="Epochs")
            save_interval = gr.Slider(1, 100, value=10, label="Save every X epochs")
            complexity = gr.Slider(0.1, 1.0, value=0.5, label="Training Complexity (Speed vs Accuracy)")

            batch_size = gr.Slider(1, 128, value=32, label="Batch Size")
            train_button = gr.Button("Train")
            output_train = gr.Textbox(label="Training Status")

            train_button.click(
                fn=train_model_ui,
                inputs=[model_name, epochs, save_interval, complexity, batch_size, device_id],
                outputs=output_train
            )

        # PARAMETERS TAB
        with gr.Tab("Parameters"):
            gr.Markdown("## ⚙️ Model Parameters with 3D Visualization")

            model_param_select = gr.Dropdown(
                choices=[m for m in os.listdir(MODELS_DIR) if m.endswith(".pth")],
                label="Select Model"
            )
            
            # Button to trigger the refresh
            refresh_button = gr.Button("Refresh Model List")
            
            # Set the refresh button to update the model dropdown when clicked
            refresh_button.click(refresh_model_dropdown)

            param_output = gr.Textbox(label="Model Parameters")

            # Dynamic plot placeholders
            plots = [gr.Plot(label=f"Layer {i + 1}") for i in range(4)]

            load_params_button = gr.Button("Load Parameters")

            def load_and_show(model):
                params, plots_data, _ = load_params_with_viz(os.path.join("./models", model))

                # Return model info and dynamic plots
                return [params] + plots_data

            load_params_button.click(
                fn=load_and_show,
                inputs=[model_param_select],
                outputs=[param_output] + plots
            )

        # Gradio interface
        with gr.Tab("Generate"):
            gr.Markdown("## 🎤 Generate Audio")

            model_select = gr.Dropdown(
                choices=[m for m in os.listdir(MODELS_DIR) if m.endswith(".pth")],
                label="Select Model"
            )

            duration = gr.Slider(1, 30, value=5, label="Duration (seconds)")
            bpm = gr.Slider(60, 180, value=120, label="BPM")
            prompt = gr.Textbox(label="Prompt", placeholder="Enter a prompt for audio generation", value="A stunning Array of Electronc sounds in syncopated rythms with eastern vibes.")
            # File input for .wav, .mp3, and .flac formats
            #file_input = gr.File(label="Upload Audio File", file_types=[".wav", ".mp3", ".flac"])
            generate_button = gr.Button("Generate")
            output_generate = gr.Textbox(label="Generated Audio Path")

            # Trigger audio generation pipeline
            generate_button.click(
                fn=generate_audio,
                inputs=[model_select, duration, bpm, prompt],
                outputs=output_generate
            )

# Launch the Gradio app
app.launch(server_name=args.listen, server_port=args.port, share=True)
