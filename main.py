from vosk import Model, KaldiRecognizer
import os

model_path = "/home/mw-user-new/Desktop/Test/NLP/App/intent_voice/vosk-model-es-0.42"

if not os.path.exists(model_path):
    raise ValueError("Download the Vosk model from https://alphacephei.com/vosk/models")

model = Model(model_path)

def get_recognizer(sample_rate=16000):
    return KaldiRecognizer(model, sample_rate)