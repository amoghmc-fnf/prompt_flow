import os
import time
from openai import AzureOpenAI
import tiktoken
import base64
from mimetypes import guess_type
import fitz


def main():
    # Start measuring time
    start_time = time.time()

    convert_pdf_to_image("HS0825664.pdf")
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

def get_image_paths_from_dir(image_dir):
    image_paths = []
    for image in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image)
        image_paths.append(image_path)

    return image_paths

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def get_extracted_content():
    # Initialize the Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01"
    )

    image_paths = get_image_paths_from_dir("outputFolder")

    # Process each image 
    ai_response = ""

    for image_path in image_paths:
        response = client.chat.completions.create(
            model="gpt-4o-mini-std",  # model = "deployment_name".
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts only titles/headings/sub-headings from the given picture and ignore everything else. Do it quickly as possible as you are looking for titles/headings/sub-headings only. You will be given a 'TITLE DOCUMENT' that is prefilled with titles or empty and u need to use it to fill any missing titles/headings/sub-headings"},
                {"role": "user", "content": "TITLE DOCUMENT:\n" + ai_response},
                { "role": "user", "content": [{ 
                        "type": "image_url",
                        "image_url": {
                            "url": local_image_to_data_url(image_path)
                        }
                    }]
                }
            ]
        )
        ai_response = response.choices[0].message.content
    
    return ai_response


if __name__ == "__main__":
    main()
