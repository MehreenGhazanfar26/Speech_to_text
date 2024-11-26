import streamlit as st
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa

# Load the model and processor (backend setup)
model_name = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Function to transcribe audio
def transcribe_audio(audio_file):
    # Load the audio file
    speech, rate = librosa.load(audio_file, sr=16000)  # Ensure 16kHz sampling rate
    # Preprocess the audio
    input_values = processor(speech, sampling_rate=rate, return_tensors="pt").input_values
    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
    # Decode the transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Streamlit User Interface (frontend setup)
st.title("Speech-to-Text App")
st.write("Upload an audio file, and this app will transcribe it into text.")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Display the uploaded audio file
    st.audio(uploaded_file, format="audio/wav")
    
    # Save the uploaded file to a temporary location
    with open("uploaded_audio.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    # Perform transcription
    st.write("Processing the audio file...")
    transcription = transcribe_audio("uploaded_audio.wav")
    
    # Display the transcription
    st.write("### Transcription:")
    st.text(transcription)

