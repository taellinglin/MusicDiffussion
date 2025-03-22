import os
import librosa
import soundfile as sf
import json
import torch
import torchaudio
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

# ROYGBIV colors
ROYGBIV = [
    Fore.RED,        # R
    Fore.YELLOW,     # O
    Fore.GREEN,      # Y
    Fore.BLUE,       # B
    Fore.MAGENTA,    # I
    Fore.CYAN,       # V
]

def color_text(text, color_index):
    """ Apply ROYGBIV color cycling """
    return ROYGBIV[color_index % len(ROYGBIV)] + text + Style.RESET_ALL

def save_dataset(dataset, output_file):
    """ Save dataset to a JSON file """
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(color_text(f"✅ Saved dataset to {output_file}", 3))

def save_sample(y, sr, sample_name, samples_folder):
    """ Save short audio files (samples) in a separate folder """
    sample_file = os.path.join(samples_folder, f"{sample_name}.wav")
    sf.write(sample_file, y, sr)
    print(color_text(f"Sample saved: {sample_file}", 2))

def extract_features(audio_path, sample_rate=44100):
    """ Extract features using torchaudio and librosa """
    # Load audio file
    waveform, sr = torchaudio.load(audio_path, normalize=True)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    
    # Extract features: MFCC, Spectrogram, Chroma
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sample_rate, 
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    )(waveform)

    # Extract a chromagram
    chroma = torchaudio.transforms.Chroma()(waveform)

    # Convert to numpy arrays
    mfcc = mfcc.mean(dim=-1).numpy()  # Mean across time frames
    chroma = chroma.mean(dim=-1).numpy()  # Mean across time frames

    return {
        "mfcc": mfcc.tolist(),
        "chroma": chroma.tolist(),
    }

def process_mp3_folder(mp3_folder, output_folder, dataset_name, sample_rate=44100, bit_depth='PCM_16', format_choice='wav', batch_size=32, sample_threshold=10):
    """
    Normalize and convert MP3 files in a folder, save every batch as a JSON,
    and categorize short files as samples.
    Args:
        mp3_folder: Path to the MP3 folder.
        output_folder: Path to save the dataset.
        dataset_name: The name of the dataset for the output file.
        sample_rate: Target sample rate.
        bit_depth: Bit depth ('PCM_16', 'PCM_24', etc.)
        format_choice: Output format ('wav', 'mp3', 'flac')
        batch_size: Number of files to process before saving the dataset.
        sample_threshold: Duration in seconds to classify a file as a sample.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Path for the main dataset and samples folder
    dataset_file = os.path.join(output_folder, f"{dataset_name}.json")
    samples_folder = os.path.join(output_folder, "samples")
    if not os.path.exists(samples_folder):
        os.makedirs(samples_folder)

    dataset = []  # Will hold the processed data
    processed_count = 0  # Counter for the processed MP3 files

    # Process each MP3 file in the folder
    for idx, file in enumerate(tqdm(os.listdir(mp3_folder), desc="Processing MP3s")):
        if file.endswith('.mp3'):
            audio_path = os.path.join(mp3_folder, file)

            # Load audio
            y, sr = librosa.load(audio_path, sr=sample_rate)

            # Check the duration to categorize as sample or normal
            duration = librosa.get_duration(y=y, sr=sr)
            audio_data = {
                "filename": file,
                "duration": duration,
                "sample_rate": sr,
            }

            if duration <= sample_threshold:
                # Save short files as samples
                save_sample(y, sr, os.path.splitext(file)[0], samples_folder)
            else:
                # Extract audio features and add to dataset
                features = extract_features(audio_path, sample_rate)
                audio_data.update(features)
                dataset.append(audio_data)
            
            # Every `batch_size` files, save the dataset to a JSON file
            processed_count += 1
            if processed_count % batch_size == 0:
                save_dataset(dataset, dataset_file)
                print(color_text(f"Saved after processing {processed_count} files.", processed_count % len(ROYGBIV)))

            print(f"Processed: {file}")

    # Final save after all files are processed
    if len(dataset) % batch_size != 0:
        save_dataset(dataset, dataset_file)
        print(color_text(f"Final dataset saved after processing {len(dataset)} files.", len(dataset) % len(ROYGBIV)))

    print(f"✅ All data processed and saved at: {dataset_file}")
