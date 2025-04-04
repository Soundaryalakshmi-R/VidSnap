import streamlit as st
import os
import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip

# Function to summarize the video (placeholder logic)
def summarize_video(input_video_path, output_video_path, summary_duration=5):
    """ Extracts a short summary clip from the input video. """
    video = VideoFileClip(input_video_path)
    duration = video.duration

    # Extract a clip from the middle of the video
    start_time = max(0, duration / 2 - summary_duration / 2)
    end_time = min(duration, start_time + summary_duration)
    
    summary_clip = video.subclip(start_time, end_time)
    summary_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

# Streamlit UI
st.title("🎥 Video Summarization App")
st.write("Upload a video, and we'll generate a summarized version!")

# File uploader
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    input_video_path = os.path.join(temp_dir, uploaded_file.name)

    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(input_video_path)

    # Generate summary
    output_video_path = os.path.join(temp_dir, "summary.mp4")
    summarize_video(input_video_path, output_video_path)

    # Display summarized video
    st.subheader("📌 Summarized Video")
    st.video(output_video_path)

    # Provide download option
    with open(output_video_path, "rb") as f:
        st.download_button("Download Summary", f, file_name="summary.mp4", mime="video/mp4")
