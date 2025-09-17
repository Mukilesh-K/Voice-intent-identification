from pydub import AudioSegment, effects
import soundfile as sf
import numpy as np
import json
import noisereduce as nr
from main import get_recognizer
import os

def mp3_to_wav(mp3_path, wav_path="temp.wav"):
    # Load MP3 and convert to mono 16kHz
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    # Normalize volume
    audio = effects.normalize(audio)
    
    # Export to WAV
    audio.export(wav_path, format="wav")
    return wav_path

def reduce_noise(audio_data, sample_rate):
    # Apply noise reduction using noisereduce
    return nr.reduce_noise(y=audio_data, sr=sample_rate)

def transcribe_audio(audio_path):
    print(f"\nProcessing audio: {audio_path}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    ext = os.path.splitext(audio_path)[-1].lower()
    if ext == ".mp3":
        wav_path = mp3_to_wav(audio_path)
    elif ext == ".wav":
        wav_path = audio_path  # already WAV
    else:
        raise ValueError("Unsupported audio format: only .mp3 and .wav allowed")

    # Load audio
    audio_data, sample_rate = sf.read(audio_path)

    # Noise reduction
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # Take one channel if stereo

    audio_data = reduce_noise(audio_data, sample_rate)

    # Recognizer setup
    recognizer = get_recognizer(sample_rate)
    text = []

    # Process in chunks
    for chunk in range(0, len(audio_data), 4000):
        chunk_data = audio_data[chunk:chunk+4000].astype(np.float32).tobytes()
        if recognizer.AcceptWaveform(chunk_data):
            result = recognizer.Result()
            text.append(json.loads(result)["text"])
    
    # Final result
    final_result = recognizer.FinalResult()
    text.append(json.loads(final_result)["text"])
    return " ".join(text)

# Example usage
wav_file = mp3_to_wav("/home/mw-user-new/Desktop/Test/NLP/App/intent_voice/289460.mp3")
transcript = transcribe_audio(wav_file)
print("Transcript:", transcript)