import os
import traceback
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import numpy as np
import PyPDF2
import requests
import tempfile
from typing import List

# --- Configuration ---
# This secure version reads keys from the environment variables on your server.
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
        GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
        logging.info("Google Gemini model configured successfully.")
    else:
        GEMINI_MODEL = None
        logging.warning("GOOGLE_API_KEY not found. Q&A functionality will be disabled.")
except Exception as e:
    logging.error(f"Failed to initialize models: {e}")
    GEMINI_MODEL = None

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

class SemanticRetriever:
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.embeddings = self._embed_chunks_via_api()

    def _embed_chunks_via_api(self) -> np.ndarray:
        if not self.chunks:
            return None
        logging.info(f"Getting embeddings for {len(self.chunks)} chunks via API...")
        try:
            result = genai.embed_content(
                model='models/embedding-001',
                content=self.chunks,
                task_type="RETRIEVAL_DOCUMENT"
            )
            logging.info("API embeddings received successfully.")
            return np.array(result['embedding'])
        except Exception as e:
            logging.error(f"Failed to get embeddings from API: {e}")
            return None

    def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        if self.embeddings is None:
            return "Error: Document embeddings are not available."
        try:
            result = genai.embed_content(
                model='models/embedding-001',
                content=query,
                task_type="RETRIEVAL_QUERY"
            )
            query_embedding = np.array(result['embedding'])
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            relevant_context = "\n---\n".join([self.chunks[i] for i in top_indices])
            return relevant_context
        except Exception as e:
            logging.error(f"Failed to get query embedding or perform search: {e}")
            return f"Error during search: {e}"

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
    # This check is now simpler and correct
    if not GEMINI_MODEL:
        return jsonify({'error': 'System not initialized. LLM model is not loaded.'}), 503
    
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

        retriever = SemanticRetriever(document_chunks)
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
    return jsonify({
        'status': 'healthy' if GEMINI_MODEL else 'degraded',
        'models_loaded': {
            'gemini': 'available' if GEMINI_MODEL else 'unavailable'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
