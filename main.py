import os
import json
import pdfplumber
import fitz
from collections import defaultdict
from typing import Dict, List
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import pipeline
from pathlib import Path
import concurrent.futures
import unicodedata
import logging
import warnings

# Cache and warnings setup
os.environ['HF_HOME'] = '/app/cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/app/cache/huggingface/transformers'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load models with pre-trained zero-shot for better heading detection
try:
    sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    logging.info("Loaded SentenceTransformer")
except Exception as e:
    logging.warning(f"SentenceTransformer failed: {e}")
    sentence_model = None

try:
    nlp_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    logging.info("Loaded zero-shot classifier for headings")
except Exception as e:
    logging.warning(f"Zero-shot failed: {e}. Using heuristic.")
    def fallback_classifier(text):
        score = 0.8 if any(k in text.lower() for k in ['section', 'chapter', 'appendix', 'title']) or len(text.split()) < 10 else 0.4
        return {'scores': [score], 'labels': ['heading' if score > 0.5 else 'text']}
    nlp_classifier = lambda text: fallback_classifier(text)

def extract_title_and_headings(pdf_path: str) -> Dict:
    logging.info(f"Processing: {pdf_path}")
    doc = fitz.open(pdf_path)
    pdf = pdfplumber.open(pdf_path)
    
    font_sizes = defaultdict(int)
    for page in pdf.pages:
        for char in page.chars:
            font_sizes[round(char['size'], 2)] += 1
    
    if not font_sizes:
        pdf.close()
        doc.close()
        return {"title": "", "outline": []}
    
    sizes_array = np.array(list(font_sizes.keys())).reshape(-1, 1)
    num_clusters = min(4, len(sizes_array))
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(sizes_array)
    cluster_means = [sizes_array[kmeans.labels_ == c].mean() for c in set(kmeans.labels_)]
    sorted_sizes = sorted(cluster_means, reverse=True)
    heading_levels = {sorted_sizes[i]: f"H{i+1}" for i in range(len(sorted_sizes))}
    
    # Enhanced title extraction
    first_page = pdf.pages[0]
    title_candidates = []
    current_line = []
    prev_top = None
    for char in sorted(first_page.chars, key=lambda c: (c['top'], c['x0'])):
        size = round(char['size'], 2)
        if size == max(font_sizes.keys()):
            normalized = unicodedata.normalize('NFKC', char['text'])  # Stronger for multilingual/garble
            if prev_top is None or abs(char['top'] - prev_top) < 6:
                current_line.append(normalized)
            else:
                line_text = ''.join(current_line).strip().replace('  ', ' ')  # Fix doubles
                if line_text:
                    title_candidates.append(line_text)
                current_line = [normalized]
            prev_top = char['top']
    if current_line:
        line_text = ''.join(current_line).strip().replace('  ', ' ')
        if line_text:
            title_candidates.append(line_text)
    title = ' '.join(title_candidates) if title_candidates else doc[0].get_text().strip().split('\n')[0]
    logging.info(f"Title: {title}")
    
    # Parallel heading extraction with tuned filters
    outline = []
    seen_embeddings = []
    def process_page(page_num, page):
        local_outline = []
        page_height = page.height
        lines = defaultdict(list)
        for char in page.chars:
            if 0.1 * page_height < char['top'] < 0.9 * page_height:  # Loosened for more recall
                lines[round(char['top'])].append(char)
        
        for line_top in sorted(lines.keys()):
            line_chars = lines[line_top]
            if not line_chars:
                continue
            line_font_sizes = defaultdict(int)
            for c in line_chars:
                line_font_sizes[round(c['size'], 2)] += 1
            main_size = max(line_font_sizes, key=line_font_sizes.get)
            cluster_mean = kmeans.predict([[main_size]])[0]
            level_size = cluster_means[cluster_mean]
            if level_size not in heading_levels:
                continue
            
            text = ''.join(unicodedata.normalize('NFKC', c['text']) for c in sorted(line_chars, key=lambda x: x['x0'])).strip().replace('  ', ' ')
            if not text or len(text) < 3:
                continue
            
            # Zero-shot classification for "heading" likelihood
            result = nlp_classifier(text, candidate_labels=["heading", "text"])
            if result['scores'][0] < 0.5:  # Lower threshold for recall
                continue
            
            if sentence_model:
                embedding = sentence_model.encode(text)
                if any(util.pytorch_cos_sim(torch.tensor(embedding), torch.tensor(seen)).item() > 0.75 for seen in seen_embeddings):
                    continue
                seen_embeddings.append(embedding)
            
            local_outline.append({"level": heading_levels[level_size], "text": text, "page": page_num})
        return local_outline
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_page, i+1, page) for i, page in enumerate(pdf.pages)]
        for future in concurrent.futures.as_completed(futures):
            outline.extend(future.result())
    
    pdf.close()
    doc.close()
    logging.info(f"Extracted {len(outline)} items")
    return {"title": title, "outline": outline}

def process_pdfs():
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(input_dir.glob("*.pdf"))
    for pdf_file in pdf_files:
        result = extract_title_and_headings(str(pdf_file))
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logging.info(f"Processed {pdf_file.name}")

if __name__ == "__main__":
    logging.info("Starting processing")
    process_pdfs()
    logging.info("Completed processing")
