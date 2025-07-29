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
        # Use Pro for better accuracy while staying memory efficient
        GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-pro')
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

class MemoryEfficientRetriever:
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.embeddings = None
        self._generate_embeddings()

    def _generate_embeddings(self):
        """Generate embeddings in small batches to save memory"""
        if not self.chunks:
            return
        
        logging.info(f"Generating embeddings for {len(self.chunks)} chunks...")
        try:
            # Process in very small batches to save memory
            batch_size = 20  # Reduced from 100
            all_embeddings = []
            
            for i in range(0, len(self.chunks), batch_size):
                batch = self.chunks[i:i+batch_size]
                result = genai.embed_content(
                    model='models/embedding-001',  # Use older model - more memory efficient
                    content=batch,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                
                if isinstance(result['embedding'][0], list):
                    all_embeddings.extend(result['embedding'])
                else:
                    all_embeddings.append(result['embedding'])
                
                # Force garbage collection after each batch
                gc.collect()
            
            self.embeddings = np.array(all_embeddings, dtype=np.float32)  # Use float32 instead of float64
            logging.info("Embeddings generated successfully")
        except Exception as e:
            logging.error(f"Failed to generate embeddings: {e}")
            self.embeddings = None

    def get_relevant_context(self, query: str, top_k: int = 3) -> str:
        if self.embeddings is None:
            return "Error: Document embeddings not available."
        
        try:
            result = genai.embed_content(
                model='models/embedding-001',
                content=query,
                task_type="RETRIEVAL_QUERY"
            )
            query_embedding = np.array(result['embedding'], dtype=np.float32)
            
            # Compute similarities efficiently
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            relevant_chunks = [self.chunks[i] for i in top_indices if similarities[i] > 0.2]
            
            if not relevant_chunks:
                return "No relevant information found."
            
            return "\n---\n".join(relevant_chunks)
        except Exception as e:
            logging.error(f"Error in retrieval: {e}")
            return f"Error during search: {e}"

class OptimizedAnsweringEngine:
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
                    temperature=0.05,  # Even lower for more precise policy answers
                    max_output_tokens=300,  # Slightly longer for detailed policy info
                    top_p=0.8,  # Add top_p for better consistency
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
    start_time = time.time()
    initial_memory = get_memory_usage()
    
    if not GEMINI_MODEL:
        return jsonify({'error': 'System not initialized. LLM model is not loaded.'}), 503
    
    data = request.get_json()
    if not data or 'documents' not in data or 'questions' not in data:
        return jsonify({'error': 'Invalid request body. "documents" and "questions" fields are required.'}), 400
    
    doc_url = data['documents']
    questions = data['questions']

    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        return jsonify({'error': '"questions" must be a list of strings.'}), 400

    # Limit questions to prevent memory issues
    if len(questions) > 15:
        questions = questions[:15]
        logging.warning("Limited to first 15 questions for memory efficiency")

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        # Process document with memory efficiency
        processor = MemoryEfficientDocumentProcessor()
        if not processor.download_file(doc_url, tmp_path):
            return jsonify({'error': f'Failed to download document from URL: {doc_url}'}), 500
        
        logging.info(f"Download completed. Memory: {get_memory_usage():.1f}MB")
        
        document_chunks = processor.process_pdf_streaming(tmp_path)
        if not document_chunks:
            return jsonify({'error': 'Failed to extract text from document.'}), 500

        logging.info(f"Processing completed. Memory: {get_memory_usage():.1f}MB")

        # Create retriever with memory monitoring
        retriever = MemoryEfficientRetriever(document_chunks)
        if retriever.embeddings is None:
            return jsonify({'error': 'Failed to generate document embeddings.'}), 500
            
        logging.info(f"Embeddings completed. Memory: {get_memory_usage():.1f}MB")

        answer_engine = OptimizedAnsweringEngine(GEMINI_MODEL)
        
        # Process questions with limited parallelism for memory control
        max_workers = min(3, len(questions))  # Limit concurrent threads
        answers = []
        
        # Split questions into smaller batches
        batch_size = 5
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(answer_engine.get_answer, q, retriever.get_relevant_context(q))
                    for q in batch_questions
                ]
                
                for future in futures:
                    try:
                        answer = future.result(timeout=30)
                        answers.append(answer)
                    except Exception as exc:
                        logging.error(f"Question processing error: {exc}")
                        answers.append(f"Error: {str(exc)[:50]}")
            
            # Clean up between batches
            gc.collect()
            logging.info(f"Batch {i//batch_size + 1} completed. Memory: {get_memory_usage():.1f}MB")
        
        total_time = time.time() - start_time
        final_memory = get_memory_usage()
        
        logging.info(f"All questions processed in {total_time:.2f}s. Peak memory: {final_memory:.1f}MB")
        
        return jsonify({
            'answers': answers,
            'processing_time': f"{total_time:.2f}s",
            'memory_used': f"{final_memory:.1f}MB"
        })

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'Internal server error: {str(e)[:100]}'}), 500
    
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        
        # Force garbage collection
        gc.collect()
        logging.info("Cleanup completed")

@app.route('/api/health', methods=['GET'])
def health_check():
    memory_usage = get_memory_usage()
    return jsonify({
        'status': 'healthy' if GEMINI_MODEL else 'degraded',
        'memory_usage_mb': f"{memory_usage:.1f}",
        'models_loaded': {
            'gemini': 'available' if GEMINI_MODEL else 'unavailable'
        }
    })

# Memory cleanup on startup
gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
