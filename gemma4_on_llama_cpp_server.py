import base64
import io
import requests
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# --- Configuration ---
SERVER_URL = "http://localhost:8080/v1/chat/completions"
AUDIO_FILE = "C:/Users/lowin/OneDrive/Desktop/messy/Recording_cantonese.wav" #"C:/Users/lowin/OneDrive/Desktop/messy/f5oga-kx2uu.wav"
CHUNK_DURATION = 30  # seconds
OVERLAP_DURATION = 1  # second
SAMPLE_RATE = 16000

def encode_audio_segment(audio_np):
    """Converts numpy audio segment to base64 WAV string."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, SAMPLE_RATE, format='WAV')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def transcribe_segment(audio_segment):
    """Sends a single chunk to the llama-server via HTTP POST."""
    audio_b64 = encode_audio_segment(audio_segment)
    
    payload = {
        "model": "gemma-4-e4b-it",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio", 
                        "input_audio": {"data": audio_b64, "format": "wav"}
                    },
                    {
                        "type": "text", 
                        "text": "Transcribe the audio accurately. Provide only the text."
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "stream": False
    }

    try:
        response = requests.post(SERVER_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error during API call: {e}")
        return ""

def process_long_audio(file_path):
    # Load audio and prepare parameters
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    
    total_samples = len(audio)
    chunk_samples = CHUNK_DURATION * SAMPLE_RATE
    overlap_samples = OVERLAP_DURATION * SAMPLE_RATE
    stride = chunk_samples - overlap_samples
    
    full_transcript = []
    
    # Sliding window processing
    print(f"🚀 Sending chunks to server at {SERVER_URL}...")
    for start in tqdm(range(0, total_samples, stride)):
        end = start + chunk_samples
        segment = audio[start:end]
        
        # Skip segments shorter than 0.5 seconds
        if len(segment) < (SAMPLE_RATE / 2):
            break
            
        text = transcribe_segment(segment)
        if text:
            full_transcript.append(text)

    return " ".join(full_transcript)

if __name__ == "__main__":
    final_text = process_long_audio(AUDIO_FILE)
    
    print("\n--- Transcription Result ---")
    print(final_text)
    
    # Save output
    with open("output_transcription.txt", "w", encoding="utf-8") as f:
        f.write(final_text)