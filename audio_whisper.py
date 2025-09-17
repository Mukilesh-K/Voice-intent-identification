# audio_conversion.py
from pydub import AudioSegment, effects
import soundfile as sf
import numpy as np
import noisereduce as nr
import openai
import os
import logging
from speechbrain.inference.interfaces import foreign_class

COST_PER_SECOND = 0.006 / 60  # Whisper API cost estimate per second
logger = logging.getLogger(__name__)

openai_api_key = os.getenv("OPENAI_API_KEY")  # Make sure this is set

def convert_to_wav(audio_path, wav_path="temp.wav"):
    """Convert MP3 or WAV to mono, 16kHz, 16-bit PCM WAV"""
    ext = os.path.splitext(audio_path)[-1].lower()

    # Let pydub detect format correctly
    if ext == ".mp3":
        audio = AudioSegment.from_file(audio_path, format="mp3")
    else:
        audio = AudioSegment.from_file(audio_path)  # autodetect .wav, .ogg, etc.

    audio = audio.set_channels(1).set_frame_rate(16000)
    audio = effects.normalize(audio)
    
    # Export to 16-bit PCM WAV
    audio.export(wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])
    return wav_path

def reduce_noise(audio_data, sample_rate):
    return nr.reduce_noise(y=audio_data, sr=sample_rate)

def preprocess_audio(audio_path, output_path="processed.wav"):
    """Apply noise reduction and write cleaned audio"""
    data, sample_rate = sf.read(audio_path)
    if len(data.shape) > 1:
        data = data[:, 0]  # Convert stereo to mono if needed
    reduced = reduce_noise(data, sample_rate)
    sf.write(output_path, reduced, sample_rate)
    return output_path

def calculate_cost(duration_in_seconds):
    return duration_in_seconds * COST_PER_SECOND

def detect_emotion(audio_path):
    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier"
    )
    _, _, _, label = classifier.classify_file(audio_path)
    return label

def transcribe_audio(audio_path):
    print(f"\nProcessing audio: {audio_path}")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    try:
        # Convert to 16kHz mono WAV
        wav_path = convert_to_wav(audio_path)
        processed_path = preprocess_audio(wav_path)

        # Transcribe with Whisper
        client = openai.OpenAI(api_key=openai_api_key)
        with open(processed_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
                language="en"
            )

        # Detect emotion
        emotion_label = detect_emotion(processed_path)

        return transcript, emotion_label

    finally:
        # Clean up temp files regardless of success/failure
        for path in [locals().get('wav_path'), locals().get('processed_path')]:
            if path and os.path.exists(path):
                os.remove(path)

# Run independently for testing
if __name__ == "__main__":
    test_path = "/home/mw-user-new/Desktop/Test/NLP/App/intent_voice/289460.mp3"  # or .wav
    transcript, emotion = transcribe_audio(test_path)
    print("\nTranscript:", transcript)
    print("Detected Emotion:", emotion)
