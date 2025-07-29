import os
import traceback
import logging
import faiss  # Import FAISS
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import requests
import tempfile
from typing import List

# --- Configuration ---
# FOR LOCAL TESTING ONLY. DO NOT SHARE THIS CODE WITH KEYS IN IT.
# ---
# Replace with your actual Google API Key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
HACKRX_API_KEY = os.getenv('HACKRX_API_KEY')
# ---

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Models (Load Once) ---
try:
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
        logging.info("Google Gemini model configured successfully.")
    else:
        GEMINI_MODEL = None
        logging.warning("GOOGLE_API_KEY not found. Q&A functionality will be disabled.")
    
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("SentenceTransformer embedding model loaded successfully.")

except Exception as e:
    logging.error(f"Failed to initialize models: {e}")
    GEMINI_MODEL = None
    EMBEDDING_MODEL = None

# --- Core Logic Classes ---

class DocumentProcessor:
    def download_file(self, url: str, target_path: str) -> bool:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"Successfully downloaded file from {url}")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download file from {url}: {e}")
            return False

    def process_pdf(self, file_path: str) -> List[str]:
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        chunks.append(f"[Page {i+1}] {text.strip()}")
            logging.info(f"Processed {len(chunks)} pages from PDF: {os.path.basename(file_path)}")
        except Exception as e:
            logging.error(f"Error processing PDF {file_path}: {e}")
        return chunks

# -----------------------------------------------------------------------------
# ------------------ UPDATED SEMANTIC RETRIEVER WITH FAISS --------------------
# -----------------------------------------------------------------------------
class SemanticRetriever:
    """
    Creates a FAISS index for document chunks and retrieves relevant chunks for a query.
    """
    def __init__(self, chunks: List[str], model):
        self.chunks = chunks
        self.model = model
        self.index = None
        self.embeddings = self._embed_chunks()
        if self.embeddings is not None:
            self._build_faiss_index()

    def _embed_chunks(self) -> np.ndarray:
        """Creates embeddings for all document chunks."""
        if not self.chunks or self.model is None:
            return None
        logging.info(f"Creating embeddings for {len(self.chunks)} chunks...")
        embeddings = self.model.encode(self.chunks, show_progress_bar=False)
        logging.info("Embeddings created successfully.")
        return embeddings.astype('float32') # FAISS requires float32

    def _build_faiss_index(self):
        """Builds a FAISS index from the document embeddings."""
        if self.embeddings is None:
            logging.warning("Embeddings are not available, skipping FAISS index build.")
            return
            
        dimension = self.embeddings.shape[1]
        logging.info(f"Building FAISS index with dimension {dimension}...")
        
        # Using IndexFlatL2 for simple Euclidean distance search.
        # IndexFlatIP for dot product (cosine similarity) is also a good choice.
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        logging.info(f"FAISS index built successfully with {self.index.ntotal} vectors.")

    def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """Finds the most relevant chunks for a given query using FAISS."""
        if self.index is None:
            return "Error: FAISS index is not available."
            
        query_embedding = self.model.encode([query]).astype('float32')
        
        # Search the FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve the actual chunk text using the indices
        relevant_context = "\n---\n".join([self.chunks[i] for i in indices[0] if i != -1])
        return relevant_context
# -----------------------------------------------------------------------------
# ------------------------- END OF UPDATED SECTION ----------------------------
# -----------------------------------------------------------------------------

class AnsweringEngine:
    def __init__(self, model):
        self.model = model

    def get_answer(self, question: str, context: str) -> str:
        if not self.model:
            return "Error: LLM model is not available."
        prompt = f"""
        You are an expert Q&A system. Your task is to answer the user's question based *only* on the provided context.
        If the answer is not found in the context, state that clearly. Be concise and accurate.

        CONTEXT:
        ---
        {context}
        ---

        QUESTION:
        {question}

        ANSWER:
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logging.error(f"Error during Gemini API call: {e}")
            return f"Error: Could not generate answer. Details: {e}"

# --- API Endpoints ---
@app.before_request
def check_authentication():
    # ... (This function remains the same)
    pass # Placeholder for brevity

@app.route('/hackrx/run', methods=['POST'])
def run_hackrx():
    # ... (This function remains the same)
    if not GEMINI_MODEL or not EMBEDDING_MODEL:
        return jsonify({'error': 'System not initialized. Models are not loaded.'}), 503
    data = request.get_json()
    if not data or 'documents' not in data or 'questions' not in data:
        return jsonify({'error': 'Invalid request body. "documents" and "questions" fields are required.'}), 400
    doc_url = data['documents']
    questions = data['questions']
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        return jsonify({'error': '"questions" must be a list of strings.'}), 400
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        processor = DocumentProcessor()
        if not processor.download_file(doc_url, tmp_path):
            return jsonify({'error': f'Failed to download or access the document from URL: {doc_url}'}), 500
        document_chunks = processor.process_pdf(tmp_path)
        if not document_chunks:
            return jsonify({'error': 'Failed to extract any text from the provided document.'}), 500
        retriever = SemanticRetriever(document_chunks, EMBEDDING_MODEL)
        answer_engine = AnsweringEngine(GEMINI_MODEL)
        answers = []
        for question in questions:
            logging.info(f"Processing question: '{question}'")
            context = retriever.get_relevant_context(question)
            answer = answer_engine.get_answer(question, context)
            answers.append(answer)
        logging.info("Successfully answered all questions.")
        return jsonify({'answers': answers})
    except Exception as e:
        logging.error(f"An unexpected error occurred during processing: {e}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'An internal server error occurred: {e}'}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logging.info(f"Cleaned up temporary file: {tmp_path}")
        
@app.route('/api/health', methods=['GET'])
def health_check():
    # ... (This function remains the same)
    return jsonify({
        'status': 'healthy' if GEMINI_MODEL and EMBEDDING_MODEL else 'degraded',
        'models_loaded': {
            'gemini': 'available' if GEMINI_MODEL else 'unavailable',
            'embedding': 'available' if EMBEDDING_MODEL else 'unavailable'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)