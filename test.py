# demo.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def classify_with_gpt(transcript):
    prompt_template = PromptTemplate.from_template("""
    You are an advanced call analyzer. Given the transcript of a call, perform the following tasks:

    1. **Call Classification** (choose one):  
    - Success: Customer agrees or shows clear interest.  
    - Rejected: Customer declines or shows no interest.  
    - N/A: Call is too short or lacks meaningful conversation.

    2. **Sentiment Analysis**:  
    Evaluate the sentiment of the customer's words: Positive, Neutral, or Negative.

    3. **Behavioral Suggestion**:  
    Provide a concise suggestion for the agent to improve their approach or follow up.

    Format your output as JSON like this:
    {{
    "classification": "...",
    "sentiment": "...",
    "suggestion": "..."
    }}

    Transcript:
    "{transcript}"
    """)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=openai_api_key)
    chain = prompt_template | llm
    response = chain.invoke({"transcript": transcript})
    output = response.content

    # Optionally parse the JSON safely
    import json
    try:
        parsed = json.loads(output)
        classification = parsed.get("classification")
        sentiment = parsed.get("sentiment")
        suggestion = parsed.get("suggestion")
    except json.JSONDecodeError:
        classification = sentiment = suggestion = "ParseError"

    return response.content

# Run this after getting transcript
if __name__ == "__main__":
    from audio_whisper import transcribe_audio
    path = "/home/mw-user-new/Desktop/Test/NLP/App/intent_voice/289460.mp3"
    transcript = transcribe_audio(path)
    intent = classify_with_gpt(transcript)

    print("Transcript:", transcript)
    print("Intent:", intent)
