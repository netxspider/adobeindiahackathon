import os
import json
import pdfplumber
import fitz  # PyMuPDF for handling scanned PDFs and fallback text extraction
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np
from sklearn.cluster import KMeans  # For potential table structure analysis
from sentence_transformers import SentenceTransformer, util  # For semantic table caption detection
import torch
from transformers import pipeline  # For NLP-based table description generation
from pathlib import Path
import concurrent.futures  # For parallel page processing
import unicodedata  # For multilingual text normalization
import logging
import warnings

# Setup for cache, warnings, and logging (building on 1a learnings)
os.environ['HF_HOME'] = '/app/cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/app/cache/huggingface/transformers'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load ML models for semantic analysis and description (multilingual)
try:
    sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    logging.info("Loaded SentenceTransformer for caption detection")
except Exception as e:
    logging.warning(f"SentenceTransformer failed: {e}")
    sentence_model = None

try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # For generating table descriptions
    logging.info("Loaded summarization pipeline for table insights")
except Exception as e:
    logging.warning(f"Summarizer failed: {e}. Using basic description.")
    def fallback_summarizer(text):
        return f"Table summary: {text[:100]}..." if text else "Empty table"
    summarizer = lambda text: [{'summary_text': fallback_summarizer(text)}]

def extract_tables_from_pdf(pdf_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extracts tables from a PDF: Accurate, efficient, multilingual.
    - Uses pdfplumber for primary table detection.
    - Fallback to PyMuPDF for scanned PDFs (basic row/col extraction).
    - ML: Detects captions semantically, generates summaries.
    - Output: JSON with tables per page, including cells, caption, summary.
    - Efficiency: Parallel page processing, O(p * t) where p=pages, t=tables.
    """
    logging.info(f"Processing PDF for tables: {pdf_path}")
    doc = fitz.open(pdf_path)
    pdf = pdfplumber.open(pdf_path)
    
    tables_data = defaultdict(list)  # {page_num: [table_dicts]}
    seen_captions = []  # For dedup using embeddings
    
    def process_page(page_num: int, page) -> List[Dict[str, Any]]:
        local_tables = []
        try:
            # Primary extraction with pdfplumber
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                # Normalize cells for multilingual
                normalized_table = [[unicodedata.normalize('NFKC', cell) if cell else '' for cell in row] for row in table]
                
                # Detect caption: Look for text above/below table (semantic check)
                caption = ''
                page_text = page.extract_text()
                if sentence_model and page_text:
                    sentences = page_text.split('\n')
                    for sent in sentences:
                        if len(sent) > 5:
                            embedding = sentence_model.encode(sent)
                            if any(util.pytorch_cos_sim(torch.tensor(embedding), torch.tensor(seen)).item() > 0.8 for seen in seen_captions):
                                continue
                            # Simple heuristic + ML: If looks like caption (e.g., "Table 1")
                            if 'table' in sent.lower() or any(c.isdigit() for c in sent):
                                caption = sent.strip()
                                seen_captions.append(embedding)
                                break
                
                # Generate summary using NLP
                flat_text = ' '.join([' '.join(row) for row in normalized_table if any(row)])
                summary = summarizer(flat_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                
                local_tables.append({
                    "table_id": f"table_{page_num}_{table_idx}",
                    "caption": caption,
                    "summary": summary,
                    "rows": normalized_table,
                    "row_count": len(normalized_table),
                    "col_count": len(normalized_table[0]) if normalized_table else 0
                })
        except Exception as e:
            logging.warning(f"pdfplumber failed on page {page_num}: {e}. Falling back to PyMuPDF.")
            # Fallback: Basic line-based "table" detection with PyMuPDF
            page_doc = doc[page_num - 1]
            text = page_doc.get_text("text")
            lines = text.split('\n')
            if lines:
                # Simple clustering for "rows" (assume tabular if aligned)
                faux_table = [line.split() for line in lines if line.strip()]
                flat_text = ' '.join([' '.join(row) for row in faux_table])
                summary = summarizer(flat_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                local_tables.append({
                    "table_id": f"fallback_table_{page_num}",
                    "caption": "Detected tabular content",
                    "summary": summary,
                    "rows": faux_table,
                    "row_count": len(faux_table),
                    "col_count": max(len(row) for row in faux_table) if faux_table else 0
                })
        return local_tables
    
    # Parallel processing for efficiency
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_page, i+1, page) for i, page in enumerate(pdf.pages)]
        for future in concurrent.futures.as_completed(futures):
            page_tables = future.result()
            if page_tables:
                tables_data[str(future.result()[0]['table_id'].split('_')[1])] = page_tables  # Key by page
    
    pdf.close()
    doc.close()
    logging.info(f"Extracted {sum(len(t) for t in tables_data.values())} tables")
    return dict(tables_data)  # Convert to regular dict for JSON

def process_pdfs():
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(input_dir.glob("*.pdf"))
    for pdf_file in pdf_files:
        result = extract_tables_from_pdf(str(pdf_file))
        output_file = output_dir / f"{pdf_file.stem}_tables.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logging.info(f"Processed {pdf_file.name} -> {output_file.name}")

if __name__ == "__main__":
    logging.info("Starting table extraction for Challenge 1b")
    process_pdfs()
    logging.info("Completed table extraction")
