import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load lightweight embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_text_sections(pdf_path):
    """
    Extracts page-wise text from a PDF and estimates section titles.
    Returns a list of dicts with content, title, and page number.
    """
    doc = fitz.open(pdf_path)
    sections = []

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text().strip()

        if len(text) < 50:
            continue  # Skip low-content pages

        # Heuristic: First line or most prominent line as section title
        lines = text.split("\n")
        title = lines[0].strip()[:100] if lines else f"Page {page_number+1}"

        sections.append({
            "document": os.path.basename(pdf_path),
            "page_number": page_number + 1,
            "section_title": title,
            "content": text
        })

    return sections


def embed_text(text):
    """
    Generates embedding for a given text using the shared model.
    """
    return model.encode(text)


def compute_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.
    """
    return cosine_similarity([vec1], [vec2])[0][0]
