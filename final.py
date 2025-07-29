import os
import traceback
import logging
import gc
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import numpy as np
import PyPDF2
import requests
import tempfile
from typing import List
import time

# --- Configuration ---
# FOR LOCAL TESTING ONLY. DO NOT SHARE THIS CODE WITH KEYS IN IT.
# ---
# Replace with your actual Google API Key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
HACKRX_API_KEY = os.getenv('HACKRX_API_KEY')

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Models (Load Once) ---
try:
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        # Use flash for memory efficiency, but with better config
        GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
        logging.info("Google Gemini model configured successfully.")
    else:
        GEMINI_MODEL = None
        logging.warning("GOOGLE_API_KEY not found. Q&A functionality will be disabled.")
except Exception as e:
    logging.error(f"Failed to initialize models: {e}")
    GEMINI_MODEL = None

# --- Memory-Optimized Core Logic Classes ---

class MemoryEfficientDocumentProcessor:
    def download_file(self, url: str, target_path: str) -> bool:
        try:
            # Use stream with smaller chunk size to reduce memory usage
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=4096):  # Smaller chunks
                    f.write(chunk)
            logging.info(f"Successfully downloaded file from {url}")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download file from {url}: {e}")
            return False

    def process_pdf_streaming(self, file_path: str) -> List[str]:
        """Memory-efficient PDF processing that yields chunks instead of storing all in memory"""
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        # Clean and split text more aggressively to save memory
                        cleaned_text = ' '.join(text.strip().split())
                        
                        # Smaller chunks (300 words) to reduce memory footprint
                        words = cleaned_text.split()
                        for j in range(0, len(words), 300):
                            chunk = ' '.join(words[j:j+300])
                            if chunk.strip():
                                chunks.append(f"[P{i+1}S{j//300+1}] {chunk}")
                    
                    # Force garbage collection every 5 pages
                    if (i + 1) % 5 == 0:
                        gc.collect()
                        
                    # Limit total chunks to prevent memory overflow (max ~100 chunks)
                    if len(chunks) >= 100:
                        logging.warning(f"Limiting chunks to 100 for memory efficiency")
                        break
                        
            logging.info(f"Processed {len(chunks)} chunks from {total_pages} pages")
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
            return "Error: Model not available."
        
        # Shorter, more focused prompt to reduce token usage
        prompt = f"""Based ONLY on this policy context, answer the question precisely:

CONTEXT:
{context[:2000]}  # Limit context length to save memory

QUESTION: {question}

Answer with exact policy details (amounts, periods, conditions). If not in context, say "Not specified in policy document."

ANSWER:"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=200,  # Shorter responses
                )
            )
            return response.text.strip()
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            return f"Error: {str(e)[:100]}"  # Truncate error messages

    def process_question_batch(self, questions: List[str], retriever: MemoryEfficientRetriever) -> List[str]:
        """Process questions in batch to reduce API calls"""
        answers = []
        for question in questions:
            context = retriever.get_relevant_context(question)
            answer = self.get_answer(question, context)
            answers.append(answer)
            # Clean up after each question
            gc.collect()
        return answers

# --- Memory Monitoring ---
def get_memory_usage():
    """Simple memory monitoring"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        return 0

# --- API Endpoints ---
@app.before_request
def check_authentication():
    if request.endpoint == 'run_hackrx':
        if not HACKRX_API_KEY:
            logging.warning("HACKRX_API_KEY not set. Allowing request without authentication.")
            return
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authorization header is missing or does not contain a Bearer token'}), 401
        token = auth_header.split(' ')[1]
        if token != HACKRX_API_KEY:
            return jsonify({'error': 'Invalid API Key'}), 403

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
        # Process document with memory efficiency
        processor = MemoryEfficientDocumentProcessor()
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
        logging.error(f"Unexpected error: {e}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'An internal server error occurred: {e}'}), 500
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        
        # Force garbage collection
        gc.collect()
        logging.info("Cleanup completed")

@app.route('/api/health', methods=['GET'])
def health_check():
    # ... (This function remains the same)
    return jsonify({
        'status': 'healthy' if GEMINI_MODEL and EMBEDDING_MODEL else 'degraded',
        'models_loaded': {
            'gemini': 'available' if GEMINI_MODEL else 'unavailable'
        }
    })

# Memory cleanup on startup
gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
