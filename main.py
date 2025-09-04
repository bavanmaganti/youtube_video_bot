
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from openai import OpenAI
from dotenv import load_dotenv
import os


# ------------------- LOAD ENV VARIABLES -------------------
# Load API key from .env file into environment variables
load_dotenv()

# Get OpenAI API key securely from environment
api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI client
client = OpenAI(api_key=api_key)


# ------------------- YOUTUBE VIDEO URL -------------------
VIDEO_URL = input("Enter YouTube video URL: ")


# ------------------- FETCH VIDEO METADATA -------------------
try:
    yt = YouTube(VIDEO_URL)
    print("Title:", yt.title)
    print("Description:", yt.description[:200], "...")  # Print first 200 chars
    print("Views:", yt.views)
except Exception as e:
    print("Video metadata not available:", e)


# ------------------- FETCH TRANSCRIPT -------------------
video_id = VIDEO_URL.split("v=")[-1]  # Extracts the part after "v=" in URL
transcript_text = ""

try:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = " ".join([entry['text'] for entry in transcript])  # Join transcript lines
    print("\nTranscript successfully fetched!")
except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
    print("\nTranscript: No data available")
except Exception as e:
    print("\nError fetching transcript:", e)


# ------------------- INTERACT WITH LLM -------------------
if transcript_text:
    print("\nYou can now ask questions about the transcript.")
    print("Type 'exit' to quit.\n")

    while True:
        # Take user input
        user_question = input("Your question: ")

        if user_question.lower() == "exit":
            print("Exiting chat. Goodbye!")
            break

        # Send transcript + question to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # small + cost-efficient model
            messages=[
                {"role": "system", "content": "You are an assistant that answers questions about a YouTube video transcript."},
                {"role": "user", "content": f"Transcript:\n{transcript_text}\n\nQuestion: {user_question}"}
            ],
            temperature=0.5,
        )

        # Extract and print the model's answer
        answer = response.choices[0].message.content
        print("\nAssistant:", answer, "\n")
else:
    print('No Transcrip found about video')

