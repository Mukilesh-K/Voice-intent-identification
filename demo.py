from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

openai_api_key = os.getenv("OPENAI_API_KEY")  # Set your API key in environment

def classify_with_gpt(transcript):
    # Define the prompt template
    prompt_template = PromptTemplate.from_template("""
    Clasifica esta transcripción de llamada en español en "Success", "Rejected" o "N/A":
    - Success: Cliente acepta, muestra interés.
    - Rejected: Cliente rechaza o no está interesado.
    - N/A: Llamada corta, sin conversación útil.

    Transcripción: "{transcript}"

    Clasificación:
    """)
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=openai_api_key)
    
    # Create chain
    chain = prompt_template | llm
    
    # Invoke with the transcript
    response = chain.invoke({"transcript": transcript})
    return response.content

# Example usage
if __name__ == "__main__":
    from audio_conversion import transcript  # Import from audio.py
    intent = classify_with_gpt(transcript)
    print("transcript:", transcript)
    print("Intent:", intent)