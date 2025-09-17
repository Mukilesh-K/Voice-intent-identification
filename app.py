# app.py
import streamlit as st
import os
import pandas as pd
from audio_whisper import transcribe_audio
from test import classify_with_gpt
import tempfile
from datetime import datetime
import librosa
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set page config
st.set_page_config(page_title="Audio Classification", page_icon="🎙️", layout="wide")

# Title and description
# st.title("Audio Classification System")
st.markdown("<h1 style='text-align: center;'>Audio Classification System</h1>", unsafe_allow_html=True)
# st.markdown("""
# Upload multiple MP3 files to:
# 1. Convert to WAV
# 2. Process and transcribe
# 3. Classify as Success/Rejected/N/A
# 4. Export results to CSV
# """)

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    st.markdown("Configure the processing parameters")
    
    # Language selection
    language = st.selectbox("Transcription Language", ["es", "en"], index=0)
    
    # Cost estimation
    st.markdown("### Temporary files")
    # st.markdown("Whisper API cost: $0.006 per minute")    
    
    # Clear cache button
    if st.button("Clear Temporary Files"):
        st.cache_data.clear()


# Main file uploader
uploaded_files = st.file_uploader(
    "Upload MP3 files", 
    type=["mp3", "wav"],
    accept_multiple_files=True
)

def process_files(uploaded_files):
    """Process all uploaded files and return results dataframe"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_files)
    results = []
    total_cost = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (i + 1) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Processing file {i+1} of {total_files}: {uploaded_file.name}")

            # Determine extension
            file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
            suffix = file_ext if file_ext in [".mp3", ".wav"] else ".mp3"

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp.flush()
                audio_path = tmp.name

            # Double check it exists
            if not os.path.exists(audio_path):
                st.error(f"Temporary file not found: {audio_path}")
                continue

            # Transcribe using your logic
            transcript, emotion = transcribe_audio(audio_path)

            # Classify with GPT
            import json

            raw_gpt_output = classify_with_gpt(transcript)

            try:
                gpt_data = json.loads(raw_gpt_output)
                status = gpt_data.get("classification", "N/A")
                sentiment = gpt_data.get("sentiment", "Unknown")
                suggestion = gpt_data.get("suggestion", "")
            except json.JSONDecodeError:
                status = "N/A"
                sentiment = "ParseError"
                suggestion = raw_gpt_output  # fallback to raw string

            # Get duration
            audio, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            cost = duration * (0.006 / 60)
            total_cost += cost

            # Clean up file after use
            os.unlink(audio_path)

            results.append({
                "ID": i + 1,
                "Filename": uploaded_file.name,
                "Transcript": transcript,
                "Emotion": emotion,
                "Status": status,
                "Sentiment": sentiment,
                "Suggestion": suggestion,
                "Duration": f"{duration:.2f}s"
            })


        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    # Complete processing
    progress_bar.empty()
    print(f"Processing complete! Total estimated cost: ${total_cost:.4f}")
    
    # Create dataframe from current results only
    current_results_df = pd.DataFrame(results)
    return current_results_df

# Process files when uploaded
if uploaded_files and st.button("Process Files"):
    with st.spinner("Processing files..."):
        current_results_df = process_files(uploaded_files)
        
        # Show results if any were processed
        if not current_results_df.empty:
            st.success(f"Processed {len(current_results_df)} files successfully!")
            st.dataframe(current_results_df)
            
            # Export to CSV
            csv = current_results_df.to_csv(index=False).encode('utf-8')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download Current Results as CSV",
                data=csv,
                file_name=f"audio_classification_results_{timestamp}.csv",
                mime="text/csv",
                key=f"download_{timestamp}"  # Unique key for each download button
            )

# # Instructions
# with st.expander("Instructions"):
#     st.markdown("""
#     1. Upload one or more MP3 files using the uploader above
#     2. Click "Process Files" to begin transcription and classification
#     3. View results in the table below
#     4. Download the complete results as a CSV file
    
#     **Classification Criteria:**
#     - **Success**: Client accepts, shows interest
#     - **Rejected**: Client rejects or isn't interested
#     - **N/A**: Short call, no useful conversation
    
#     **Note**: Each processing session creates a fresh CSV with only the currently uploaded files.
#     """)