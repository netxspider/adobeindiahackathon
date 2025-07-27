import fitz  # PyMuPDF
import json
import os
import sys
import numpy as np

def detect_heading_level(font_size, is_bold, position_y, page_height, z_score, text):
    """
    Advanced heuristic for heading level detection:
    - Uses font size z-score, boldness, vertical position, and text length.
    - H1: High z-score, bold, near top.
    - H2: Medium z-score, possibly bold.
    - H3: Lower z-score, above average.
    """
    if z_score > 2.0 and is_bold and position_y < page_height * 0.2:
        return "H1"
    elif z_score > 1.2 and (is_bold or len(text) < 50):
        return "H2"
    elif z_score > 0.5:
        return "H3"
    return None

def extract_outline(pdf_path, output_path, use_ocr=False):
    doc = fitz.open(pdf_path)
    outline = []
    title = doc.metadata.get("title", os.path.basename(pdf_path).replace('.pdf', ''))

    # Collect all font sizes for statistical analysis
    font_sizes = []
    text_blocks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        if use_ocr:
            # Optional OCR for scanned PDFs
            import pytesseract
            from PIL import Image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
            # Process OCR text (simplified; integrate with block extraction)
            blocks = [{"lines": [{"spans": [{"size": 10, "flags": "", "text": line}]} for line in text.split('\n')]}]
        else:
            blocks = page.get_text("dict")["blocks"]
        
        page_height = page.rect.height
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            font_sizes.append(span["size"])
                            text_blocks.append({
                                "text": text,
                                "size": span["size"],
                                "flags": span["flags"],
                                "y": span["bbox"][1],  # Top y-coordinate
                                "page_height": page_height,
                                "page_num": page_num + 1
                            })

    if not font_sizes:
        font_sizes = [10]  # Default if no text found

    # Calculate statistics for z-scores
    mean_size = np.mean(font_sizes)
    std_size = np.std(font_sizes) if np.std(font_sizes) > 0 else 1

    # Extract and filter headings
    seen = set()  # For deduplication
    for block in text_blocks:
        z_score = (block["size"] - mean_size) / std_size
        is_bold = (block["flags"] & 2) != 0 or "bold" in block.get("font", "").lower()
        level = detect_heading_level(block["size"], is_bold, block["y"], block["page_height"], z_score, block["text"])
        if level and block["text"] not in seen:
            outline.append({
                "level": level,
                "text": block["text"],
                "page": block["page_num"]
            })
            seen.add(block["text"])

    # Sort outline by page and position (approximate hierarchy)
    outline.sort(key=lambda x: x["page"])
    
    result = {
        "title": title,
        "outline": outline
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_dir = "/app/input"
    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)
    
    use_ocr = False  # Set to True if handling scanned PDFs (requires pytesseract)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            json_filename = filename.replace(".pdf", ".json")
            output_path = os.path.join(output_dir, json_filename)
            extract_outline(pdf_path, output_path, use_ocr=use_ocr)
            print(f"Processed {filename} -> {json_filename}")
