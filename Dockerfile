FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install pymupdf numpy pytesseract
# Optional: For OCR support (uncomment if needed)
RUN apt-get update && apt-get install -y tesseract-ocr
CMD ["python", "process_pdfs.py"]
