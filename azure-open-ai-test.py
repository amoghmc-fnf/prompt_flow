import os
import time
from openai import AzureOpenAI



def get_text(pdf_name: str) -> str:
    content = ""
    with open(pdf_name, 'r') as file:
        # Read all the text from the file
        content = file.read()
    return content


def chunk_text(text, max_tokens):
    tokens = text.split()  # Simple tokenization by splitting on spaces
    chunks = []
    chunk = []
    token_count = 0

    for token in tokens:
        token_length = len(token)
        if token_count + token_length + 1 > max_tokens:
            chunks.append(' '.join(chunk))
            chunk = []
            token_count = 0
        chunk.append(token)
        token_count += token_length + 1

    if chunk:
        chunks.append(' '.join(chunk))

    return chunks



def get_extracted_content():
    # Initialize the Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01"
    )
    # Example input text
    input_text = get_text("extracted_content.txt")

    # Define the maximum number of tokens per chunk
    max_tokens = 100000  # Adjust based on your model's token limit

    # Chunk the input text
    chunks = chunk_text(input_text, max_tokens)
    print(len(chunks))
    # Process each chunk
    ai_response = ""
    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-4o-mini-std",  # model = "deployment_name".
            messages=[
                {"role": "system", "content": "You are an extraction assistant that extracts titles from the content and returns a json response with key value pairs. You will be given a template that will be either empty or pre filled and you will fill/improve them as required with the latest updated content."},
                {"role": "user", "content": "TEMPLATE:\n" + ai_response},
                {"role": "user", "content": "CONTENT:\n" + chunk}
            ]
        )
        ai_response = response.choices[0].message.content
        time.sleep(60)
    
    return ai_response