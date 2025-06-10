import os
import json
import asyncio
from typing import List
from datetime import datetime
import markdown
from bs4 import BeautifulSoup

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'), override=True)
    
async def process_and_store_document(base_path: str, url: str, markdown: str):
    """Process a document and store its chunks without overwriting existing data in the CWD."""
    
    filename = os.path.join(base_path, f"{os.getenv('COLLECTION_BASE_NAME')}.json")  
    
    # Read existing data if file exists
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []  
    else:
        data = []

    data.append({
        "chunk": markdown,  # full markdown
        "url": url,
        "timestamp": datetime.now().isoformat()
    })
    
    print("Writing data to file...")
    # Write updated data back to the file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

async def crawl_parallel(base_path:str, urls: List[str], max_concurrent: int):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(base_path, url, result.markdown.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    except Exception as e:
        print("************* crawl exception ****************", str(e))
    finally:
        print("************* crawl done..... ****************")
        await crawler.close()
    
def clean_markdown(markdown_content: str) -> str:
    """Convert Markdown to plain text, removing formatting while keeping coherent text."""
    html_content = markdown.markdown(markdown_content)  # Convert Markdown to HTML
    soup = BeautifulSoup(html_content, "html.parser")  # Parse HTML with BeautifulSoup
    return soup.get_text(separator=" ", strip=True)  # Extract plain text

def process_json_files(directory: str):
    """Read each JSON file in the directory, clean the 'chunk' field, and save a new JSON file."""
    
    for filename in os.listdir(directory):
        if filename.endswith(".json") and not filename.endswith("_clean.json") and filename==f"{os.getenv('COLLECTION_BASE_NAME')}.json": 
            file_path = os.path.join(directory, filename)
            clean_file_path = os.path.join(directory, filename.replace(".json", "_clean.json"))
            
            # Read the JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)  # Load JSON list
                    if not isinstance(data, list): 
                        print(f"Skipping {filename}: JSON root is not a list")
                        continue
                except json.JSONDecodeError:
                    print(f"Skipping {filename}: Invalid JSON format")
                    continue
            
            # Process and clean each chunk
            for item in data:
                if "chunk" in item:
                    item["chunk"] = clean_markdown(item["chunk"])  # Clean the chunk text
            
            # Save the cleaned JSON
            with open(clean_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            print(f"Processed and saved: {clean_file_path}")

def write_chunks_to_txt(clean_json_path: str):
    """Reads a cleaned JSON file and writes the 'chunk' field to a text file."""
    
    # Define output TXT file name
    txt_file_path = clean_json_path.replace(".json", ".txt")

    # Read the cleaned JSON file
    if not os.path.exists(clean_json_path):
        print(f"Error: File '{clean_json_path}' not found.")
        return
    
    with open(clean_json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):  # Ensure it's a list of objects
                print(f"Error: JSON file '{clean_json_path}' does not contain a list.")
                return
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{clean_json_path}'.")
            return

    # Write the 'chunk' field to a text file, each chunk on a new line
    with open(txt_file_path, "w", encoding="utf-8") as txt_file:
        for item in data:
            if "chunk" in item:
                txt_file.write(item["chunk"] + "\n\n")  # Add newline for separation

    print(f"Chunks saved to: {txt_file_path}")
    
    return txt_file_path

async def create_pipeline_dataset(urls):
        
    script_dir = os.getcwd()  
    dataset_path = os.path.join(script_dir, 'src/ingestion_pipeline/web_site_dataset_creation/dataset')
    
    # Empty the folder
    for filename in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file
            elif os.path.isdir(file_path):
                os.rmdir(file_path)  # Remove the directory if empty
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
            
    await crawl_parallel(base_path=dataset_path, urls= urls, max_concurrent = 2)

    process_json_files(dataset_path)

    clean_json_path = os.path.join(dataset_path, f"{os.getenv('COLLECTION_BASE_NAME')}_clean.json")
    final_data_set_path = write_chunks_to_txt(clean_json_path)
    
    return final_data_set_path
