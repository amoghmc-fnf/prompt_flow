import os
import time
from openai import AzureOpenAI
import tiktoken
import fitz


def main():
    # Start measuring time
    start_time = time.time()

    output = get_extracted_content()

    # Stop measuring time
    end_time = time.time()

    # Calculate the difference
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Open a file in write mode
    with open('output.txt', 'w') as file:
        file.write(output)
    return

def convert_pdf_to_image(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page in the PDF
    for page_num in range(len(pdf_document)):
        # Get the page
        page = pdf_document.load_page(page_num)
        
        # Render the page to an image
        pix = page.get_pixmap()
        
        # Define the output image path
        output_image_path = os.path.join(output_dir, f"{page_num:03}.png")
        
        # Save the image
        pix.save(output_image_path)

    print(f"PDF has been converted to images and stored in the '{output_dir}' folder.")
    return

def get_text_from_file(txt_file: str) -> str:
    content = ""

    with open(txt_file, 'r') as file:
        # Read all the text from the file
        content = file.read()

    return content


def chunk_text(text, max_tokens):
    # Initialize the tokenizer
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    
    tokens = tokenizer.encode(text)
    chunks = []
    encoded_chunk = []
    token_count = 0

    for token in tokens:
        token_length = len(tokenizer.decode([token]))
        if token_count + token_length > max_tokens:
            # convert a list of tokens back into a human-readable string
            chunks.append(tokenizer.decode(encoded_chunk))
            encoded_chunk = []
            token_count = 0
        encoded_chunk.append(token)
        token_count += token_length

    # to account for off-by-one 
    if encoded_chunk:
        chunks.append(tokenizer.decode(encoded_chunk))

    return chunks


def get_extracted_content():
    # Initialize the Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01"
    )

    input_text = get_text_from_file("extracted_content.txt")
    system_prompt = get_text_from_file("system_prompt.txt")

    # Define the maximum number of tokens per chunk
    max_tokens = 100000  # Adjust based on your model's token limit

    # Chunk the input text
    chunks = chunk_text(input_text, max_tokens)

    # Process each chunk
    ai_response = ""

    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-4o-mini-std",  # model = "deployment_name".
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "TITLE DOCUMENT:\n" + ai_response},
                {"role": "user", "content": "TEXT DOCUMENT:\n" + chunk}
            ]
        )
        ai_response = response.choices[0].message.content
    
    return ai_response


if __name__ == "__main__":
    # main()
    convert_pdf_to_image("HS0825664.pdf", "outputFolder")