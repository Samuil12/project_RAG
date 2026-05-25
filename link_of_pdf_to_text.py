import os
import re
import cv2
import requests
import pytesseract
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pdf2image import convert_from_path

# ==========================================
# CONFIGURATION
# ==========================================
# Uncomment and set this if you are on Windows!
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

TARGET_URL = "https://www.bda.bg/images/stories/documents/bdias/B-2.htm"
PDF_DIR = "downloaded_pdfs"
OUTPUT_DIR = "extracted_texts"

# ==========================================
# 1. SCRAPING & DOWNLOADING
# ==========================================
def download_pdfs(url, limit=None):
    """Scrapes the target URL for PDF links and downloads them locally."""
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)

    print(f"Fetching links from: {url}")
    # Add a standard user-agent so the server doesn't block us
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all anchor tags ending in .pdf (case insensitive)
    pdf_links = []
    for a_tag in soup.find_all('a', href=True):
        if a_tag['href'].lower().endswith('.pdf'):
            # urljoin makes sure relative links become full URLs
            full_url = urljoin(url, a_tag['href'])
            pdf_links.append(full_url)
    
    print(f"Found {len(pdf_links)} PDF links.")
    
    # Limit for testing purposes if specified
    if limit:
        pdf_links = pdf_links[:limit]

    # Download each PDF
    for i, pdf_url in enumerate(pdf_links, 1):
        filename = pdf_url.split('/')[-1]
        filepath = os.path.join(PDF_DIR, filename)
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            print(f"[{i}/{len(pdf_links)}] Already exists: {filename}")
            continue
            
        print(f"[{i}/{len(pdf_links)}] Downloading {filename}...")
        try:
            pdf_response = requests.get(pdf_url, headers=headers)
            with open(filepath, 'wb') as f:
                f.write(pdf_response.content)
        except Exception as e:
            print(f"Failed to download {pdf_url}: {e}")

# ==========================================
# 2. IMAGE PREPROCESSING & OCR
# ==========================================
def preprocess_image(image_path):
    """Converts image to Black & White, increasing contrast for OCR."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Binarize the image. This helps ignore lighter colored artifacts.
    _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_bin

def clean_extracted_text(text):
    """Cleans up stamps, handwritten dates, and mixed English/Cyrillic units."""
    
    # 1. Clean overlapping stamp text
    stamp_phrases = [
        r"ИЗПЪЛНИТЕЛНА АГЕНЦИЯ ПО ЛЕКАРСТВАТА",
        r"РЕПУБЛИКА БЪЛГАРИЯ",
        r"НЕ ЛЕКАРСТВАТА", 
        r"АГЕНЦИЯ ПО ЛЕКАРСТВАТА",
        r"ИЗПЪЛНИТЕЛНА"
    ]
    for phrase in stamp_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)

    # 2. Remove handwritten dates (e.g., 14-04-2025)
    text = re.sub(r'\b\d{2}\s*-\s*\d{2}\s*-\s*\d{4}\b', '', text)

    # 3. Remove isolated handwritten registration numbers
    text = re.sub(r'(?m)^\s*\d{5,8}\s*$', '', text)
    text = re.sub(r'(?m)^\s*\d{3}-\d{2}\s*$', '', text)

    # 4. FIX UNITS: Force mixed alphabets (мg, кg) to English (mg, kg)
    text = re.sub(r'(?<=\d)\s*[мmМM][gгГG]\b', ' mg', text)
    text = re.sub(r'(?<=\d)\s*[кkКK][gгГG]\b', ' kg', text)
    text = re.sub(r'[СC][DД]4', 'CD4', text, flags=re.IGNORECASE)

    # 5. Clean up multiple empty lines caused by deletions
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def process_pdfs():
    """Runs OCR on all PDFs in the downloaded directory."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    
    for filename in pdf_files:
        pdf_path = os.path.join(PDF_DIR, filename)
        txt_filename = filename.replace(".pdf", ".txt")
        txt_path = os.path.join(OUTPUT_DIR, txt_filename)
        
        # Skip if we already extracted text for this PDF
        if os.path.exists(txt_path):
            print(f"Skipping OCR for {filename} (Text file already exists).")
            continue

        print(f"\nProcessing OCR for {filename}...")
        
        try:
            pages = convert_from_path(pdf_path, dpi=300)
            full_text = ""
            
            for i, page in enumerate(pages):
                print(f"  -> Extracting page {i+1}/{len(pages)}")
                temp_img_path = f"temp_page_{i}.png"
                page.save(temp_img_path, 'PNG')
                
                processed_img = preprocess_image(temp_img_path)
                
                # USING BOTH bul and eng dictionaries!
                custom_config = r'--oem 3 --psm 3 -l bul+eng'
                raw_text = pytesseract.image_to_string(processed_img, config=custom_config)
                
                cleaned_text = clean_extracted_text(raw_text)
                full_text += cleaned_text + "\n\n---PAGE BREAK---\n\n"
                
                os.remove(temp_img_path)
                
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"Saved: {txt_filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Starting pipeline...")
    # Change limit=5 to limit=None if you want to download ALL PDFs on the page
    # It is currently set to 5 so you can test it without downloading hundreds of files.
    download_pdfs(TARGET_URL, limit=5)
    process_pdfs()
    print("\nPipeline complete!")