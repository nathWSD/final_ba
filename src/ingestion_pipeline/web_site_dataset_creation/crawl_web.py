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
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []  
    else:
        data = []

    data.append({
        "chunk": markdown,  
        "url": url,
        "timestamp": datetime.now().isoformat()
    })
    
    print("Writing data to file...")
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
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
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
        
        await asyncio.gather(*[process_url(url) for url in urls])
    except Exception as e:
        print("************* crawl exception ****************", str(e))
    finally:
        print("************* crawl done..... ****************")
        await crawler.close()
    
def clean_markdown(markdown_content: str) -> str:
    """Convert Markdown to plain text, removing formatting while keeping coherent text."""
    html_content = markdown.markdown(markdown_content)  
    soup = BeautifulSoup(html_content, "html.parser")  
    return soup.get_text(separator=" ", strip=True)  

def process_json_files(directory: str):
    """Read each JSON file in the directory, clean the 'chunk' field, and save a new JSON file."""
    
    for filename in os.listdir(directory):
        if filename.endswith(".json") and not filename.endswith("_clean.json") and filename==f"{os.getenv('COLLECTION_BASE_NAME')}.json": 
            file_path = os.path.join(directory, filename)
            clean_file_path = os.path.join(directory, filename.replace(".json", "_clean.json"))
            
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f) 
                    if not isinstance(data, list): 
                        print(f"Skipping {filename}: JSON root is not a list")
                        continue
                except json.JSONDecodeError:
                    print(f"Skipping {filename}: Invalid JSON format")
                    continue
            
            for item in data:
                if "chunk" in item:
                    item["chunk"] = clean_markdown(item["chunk"])  
            
            with open(clean_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            print(f"Processed and saved: {clean_file_path}")

def write_chunks_to_txt(clean_json_path: str):
    """Reads a cleaned JSON file and writes the 'chunk' field to a text file."""
    
    txt_file_path = clean_json_path.replace(".json", ".txt")
    if not os.path.exists(clean_json_path):
        print(f"Error: File '{clean_json_path}' not found.")
        return
    
    with open(clean_json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):  
                print(f"Error: JSON file '{clean_json_path}' does not contain a list.")
                return
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{clean_json_path}'.")
            return

    with open(txt_file_path, "w", encoding="utf-8") as txt_file:
        for item in data:
            if "chunk" in item:
                txt_file.write(item["chunk"] + "\n\n")  
    print(f"Chunks saved to: {txt_file_path}")
    return txt_file_path

async def create_pipeline_dataset(urls):
        
    script_dir = os.getcwd()  
    dataset_path = os.path.join(script_dir, 'src/ingestion_pipeline/web_site_dataset_creation/dataset')
    
    for filename in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  
            elif os.path.isdir(file_path):
                os.rmdir(file_path)  
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
            
    await crawl_parallel(base_path=dataset_path, urls= urls, max_concurrent = 2)
    process_json_files(dataset_path)
    clean_json_path = os.path.join(dataset_path, f"{os.getenv('COLLECTION_BASE_NAME')}_clean.json")
    final_data_set_path = write_chunks_to_txt(clean_json_path)
    return final_data_set_path
