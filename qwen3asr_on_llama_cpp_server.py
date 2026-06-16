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
    audio_b64 = encode_audio_segment(audio_segment)
    
    # Qwen3-ASR works best with a direct, simple prompt
    payload = {
        "model": "qwen3-asr",
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
                        # Qwen3-ASR specific trigger phrase
                        "text": "<|audio_only|>Transcription:" 
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 512
    }

    try:
        response = requests.post(SERVER_URL, json=payload, timeout=60)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        # Clean up any potential model prefixes
        # content = content.replace("Transcription:", "").strip()
        # # Remove "language None" prefix if present
        # if content.startswith("language None"):
        #     content = content.replace("language None", "", 1).strip()
        return content
    except Exception as e:
        print(f"Error calling server: {e}")
        return ""

def process_long_audio(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    
    total_samples = len(audio)
    chunk_samples = CHUNK_DURATION * SAMPLE_RATE
    stride = (CHUNK_DURATION - OVERLAP_DURATION) * SAMPLE_RATE
    
    full_transcript = []
    
    print(f"🎙️ Processing with Qwen3-ASR (1.7B)...")
    for start in tqdm(range(0, total_samples, stride)):
        end = start + chunk_samples
        segment = audio[start:end]
        
        if len(segment) < (SAMPLE_RATE * 0.5): # Avoid tiny tail-end noise
            break
            
        text = transcribe_segment(segment)
        if text:
            full_transcript.append(text)

    # Note: Qwen3 is very good at handling the 1s overlap without 
    # repeating words, but a simple space-join is still the standard here.
    return " ".join(full_transcript)

if __name__ == "__main__":
    final_text = process_long_audio(AUDIO_FILE)
    
    print("\n--- Transcription Result ---")
    print(final_text)
    
    # Save output
    with open("asr_transcription.txt", "w", encoding="utf-8") as f:
        f.write(final_text)