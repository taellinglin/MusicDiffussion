import os
import librosa
import soundfile as sf

def process_mp3_folder(mp3_folder, output_folder, sample_rate=44100, bit_depth='16bit', format_choice='wav'):
    """
    Normalize and convert MP3 files in a folder.
    Args:
        mp3_folder: Path to the MP3 folder.
        output_folder: Path to save the dataset.
        sample_rate: Target sample rate.
        bit_depth: Bit depth ('16bit', '24bit', etc.)
        format_choice: Output format ('wav', 'mp3', 'flac')
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(mp3_folder):
        if file.endswith('.mp3'):
            audio_path = os.path.join(mp3_folder, file)

            # Load audio
            y, sr = librosa.load(audio_path, sr=sample_rate)

            # Normalize bit depth
            output_file = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.{format_choice}")
            
            sf.write(output_file, y, sample_rate, subtype=bit_depth)
            
            print(f"Processed: {output_file}")

    print(f"✅ Dataset saved at: {output_folder}")
