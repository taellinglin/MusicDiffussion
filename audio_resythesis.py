import librosa
import soundfile as sf
import numpy as np

def resynthesize_audio(predicted_features, output_dir='./output', sample_rate=44100, n_iter=32, hop_length=44100, target_duration=5):
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

    if np.min(predicted_features) < -100:
        mel_db = predicted_features
    else:
        mel_db = librosa.amplitude_to_db(predicted_features)

    n_frames = int(target_duration * sample_rate / hop_length)
    
    mel_db_resized = mel_db[:n_frames, :]  

    audio_waveform = librosa.feature.inverse.mel_to_audio(
        mel_db_resized, 
        sr=sample_rate, 
        hop_length=hop_length, 
        n_iter=n_iter
    )

    output_file = os.path.join(output_dir, 'resynthesized_audio.wav')
    sf.write(output_file, audio_waveform, sample_rate)

    print(f"✅ Audio saved at: {output_file}")
    return output_file
