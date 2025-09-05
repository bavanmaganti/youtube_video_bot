from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os

# ------------------- LOAD ENV VARIABLES -------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "us-east1-gcp"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


# Create Pinecone client instance
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if index exists, if not create it
if "youtube-transcripts" not in pc.list_indexes().names():
    pc.create_index(
        name="youtube-transcripts",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"   # your AWS region
        )
    )

# Connect to the index
index = pc.Index("youtube-transcripts")

# ------------------- GET YOUTUBE VIDEO -------------------
VIDEO_URL = input("Enter YouTube video URL: ")
try:
    yt = YouTube(VIDEO_URL)
    print("Title:", yt.title)
except Exception as e:
    print("Video metadata not available:", e)

video_id = VIDEO_URL.split("v=")[-1]
transcript_text = ""

try:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = " ".join([entry['text'] for entry in transcript])
    print("Transcript fetched!")
except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
    print("Transcript not available")
    exit()

# ------------------- SPLIT TRANSCRIPT INTO CHUNKS -------------------
def chunk_text(text, max_tokens=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunks.append(" ".join(words[i:i+max_tokens]))
    return chunks

chunks = chunk_text(transcript_text)

# ------------------- UPLOAD EMBEDDINGS TO PINECONE -------------------
for i, chunk in enumerate(chunks):
    embedding_resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )
    vector = embedding_resp.data[0].embedding
    index.upsert([(f"{video_id}-{i}", vector, {"text": chunk})])

print(f"{len(chunks)} chunks embedded and uploaded to Pinecone.")

# ------------------- INTERACTIVE QUERY -------------------
print("\nYou can now ask questions. Type 'exit' to quit.\n")

while True:
    user_question = input("Your question: ")
    if user_question.lower() == "exit":
        pc.delete_index('youtube-transcripts')
        break

    # Get embedding for user question
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_question
    ).data[0].embedding

    # Query Pinecone for top 3 most relevant transcript chunks
    result = index.query(vector=query_emb, top_k=3, include_metadata=True)
    context = " ".join([match['metadata']['text'] for match in result['matches']])

    # Send context + question to LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that answers questions based on YouTube video transcript chunks."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_question}"}
        ],
        temperature=0.5
    )

    answer = response.choices[0].message.content
    print("\nAssistant:", answer, "\n")
