from sentence_transformers import SentenceTransformer

# Download and save locally
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('./model/all-MiniLM-L6-v2')
