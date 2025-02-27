import os
from openai import AzureOpenAI

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01"
)



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

# Example input text
input_text = """
Does Azure OpenAI support customer managed keys? Do other Azure AI services support this too?
Your long text here...
"""

# Define the maximum number of tokens per chunk
max_tokens = 4096  # Adjust based on your model's token limit

# Chunk the input text
chunks = chunk_text(input_text, max_tokens)

# Process each chunk
for chunk in chunks:
    response = client.chat.completions.create(
        model="gpt-4o-mini-std",  # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": chunk}
        ]
    )
    print(response.choices[0].message.content)