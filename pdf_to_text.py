import os
import re
import cv2
import pytesseract
from pdf2image import convert_from_path

# Path to your PDFs
PDF_DIR = 'in_pdf'
OUTPUT_DIR = 'out_text'

def preprocess_image(image_path):
    """
    Prepares image for OCR. If your originals are in COLOR, 
    you can add color-filtering here to remove blue/red ink.
    Assuming grayscale/B&W for this example.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Increase contrast and binarize the image to make printed text pop
    _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return img_bin

def clean_extracted_text(text):
    """
    Removes known artifacts, stamp text, and handwritten patterns.
    """
    # 1. Remove specific stamp text (Agency of Medicines, Republic of Bulgaria)
    stamp_phrases = [
        r"ИЗПЪЛНИТЕЛНА АГЕНЦИЯ ПО ЛЕКАРСТВАТА",
        r"РЕПУБЛИКА БЪЛГАРИЯ",
        r"НЕ ЛЕКАРСТВАТА", # Partial stamp captures
        r"АГЕНЦИЯ ПО ЛЕКАРСТВАТА",
        r"ИЗПЪЛНИТЕЛНА"
    ]
    for phrase in stamp_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)

    # 2. Remove handwritten dates (e.g., 14-04-2025)
    text = re.sub(r'\b\d{2}\s*-\s*\d{2}\s*-\s*\d{4}\b', '', text)

    # Force "mg" and "kg" to be standard English Latin characters
    # Matches a number, optional space, and any Cyrillic/Latin mix of m, k, and g.
    text = re.sub(r'(?<=\d)\s*[мmМM][gгГG]\b', ' mg', text)
    text = re.sub(r'(?<=\d)\s*[кkКK][gгГG]\b', ' kg', text)
    
    # Optional: Ensure CD4 doesn't get read as Cyrillic 'СD4'
    text = re.sub(r'[СC][DД]4', 'CD4', text, flags=re.IGNORECASE)

    # 3. Remove isolated registration numbers (e.g., 20170345, 68523)
    # This looks for lines that are mostly just numbers
    text = re.sub(r'(?m)^\s*\d{5,8}\s*$', '', text)
    text = re.sub(r'(?m)^\s*\d{3}-\d{2}\s*$', '', text)

    # 4. Clean up multiple empty lines caused by deletions
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def process_pdfs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, filename)
            print(f"Processing {filename}...")
            
            # Convert PDF pages to images
            pages = convert_from_path(pdf_path, dpi=300)
            full_text = ""
            
            for i, page in enumerate(pages):
                # Save temp image
                temp_img_path = f"temp_page_{i}.png"
                page.save(temp_img_path, 'PNG')
                
                # Preprocess
                processed_img = preprocess_image(temp_img_path)
                
                # OCR (Ensure you have the Bulgarian lang pack installed)
                # psm 3 is fully automatic page segmentation
                #custom_config = r'--oem 3 --psm 3 -l bul'
                custom_config = r'--oem 3 --psm 3 -l bul+eng'
                raw_text = pytesseract.image_to_string(processed_img, config=custom_config)
                
                # Clean Text
                cleaned_text = clean_extracted_text(raw_text)
                full_text += cleaned_text + "\n\n---PAGE BREAK---\n\n"
                
                # Cleanup temp file
                os.remove(temp_img_path)
                
            # Save final text
            txt_filename = filename.replace(".pdf", ".txt")
            with open(os.path.join(OUTPUT_DIR, txt_filename), 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"Saved {txt_filename}")

if __name__ == "__main__":
    process_pdfs()