import os
import json
import fitz  # PyMuPDF
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

INPUT_DIR = "sample_input"
OUTPUT_FILE = "sample_output/challenge1b_output.json"

# Load lightweight model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load persona and job (hardcoded for this demo)
persona = "PhD Researcher in Computational Biology"
job_to_be_done = "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
persona_job_embedding = model.encode(persona + " " + job_to_be_done)

# Utility to extract text per page
def extract_document_sections(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text()
        if len(text.strip()) < 50:
            continue  # skip empty or low-content pages
        sections.append({
            "document": os.path.basename(pdf_path),
            "page_number": page_number + 1,
            "section_title": text.strip().split("\n")[0][:100],  # crude title guess
            "content": text.strip()
        })
    return sections

# Collect and score sections
all_sections = []
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".pdf"):
        path = os.path.join(INPUT_DIR, filename)
        sections = extract_document_sections(path)
        for section in sections:
            section["embedding"] = model.encode(section["content"])
            section["similarity"] = cosine_similarity(
                [section["embedding"]],
                [persona_job_embedding]
            )[0][0]
            all_sections.append(section)

# Sort by relevance
top_sections = sorted(all_sections, key=lambda x: x["similarity"], reverse=True)[:5]

# Compose output
output = {
    "metadata": {
        "input_documents": [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")],
        "persona": persona,
        "job_to_be_done": job_to_be_done,
        "processing_timestamp": datetime.utcnow().isoformat() + "Z"
    },
    "extracted_sections": [],
    "sub_section_analysis": []
}

# Populate output sections
for rank, sec in enumerate(top_sections, start=1):
    output["extracted_sections"].append({
        "document": sec["document"],
        "page_number": sec["page_number"],
        "section_title": sec["section_title"],
        "importance_rank": rank
    })

    # Sub-section analysis (just first 3 sentences for demo)
    refined = ". ".join(sec["content"].split(". ")[:3]) + "..."
    output["sub_section_analysis"].append({
        "document": sec["document"],
        "page_number": sec["page_number"],
        "refined_text": refined
    })

# Save JSON
os.makedirs("sample_output", exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=4)

print(f"✅ Done. Output saved to {OUTPUT_FILE}")
