import os
import traceback
import logging
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import numpy as np
import PyPDF2
import requests
import tempfile
from typing import List

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
        # Use flash model for better rate limits
        GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
        logging.info("Google Gemini model configured successfully.")
    else:
        GEMINI_MODEL = None
        logging.warning("GOOGLE_API_KEY not found. Q&A functionality will be disabled.")
except Exception as e:
    logging.error(f"Failed to initialize models: {e}")
    GEMINI_MODEL = None

# --- Rate Limiting Helper ---
class RateLimiter:
    def __init__(self, max_calls_per_minute=15):
        self.max_calls = max_calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        if len(self.calls) >= self.max_calls:
            # Wait until the oldest call is more than 1 minute old
            wait_time = 60 - (now - self.calls[0]) + 0.1  # Add small buffer
            if wait_time > 0:
                logging.info(f"Rate limit reached, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.calls = self.calls[1:]  # Remove the oldest call
        
        self.calls.append(now)

# Global rate limiter
rate_limiter = RateLimiter(max_calls_per_minute=10)  # Conservative limit

# --- Memory-Optimized Core Logic Classes ---

class MemoryEfficientDocumentProcessor:
    def download_file(self, url: str, target_path: str) -> bool:
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=4096):
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
                        cleaned_text = ' '.join(text.strip().split())
                        
                        # Larger chunks to reduce API calls (500 words instead of 300)
                        words = cleaned_text.split()
                        for j in range(0, len(words), 500):
                            chunk = ' '.join(words[j:j+500])
                            if chunk.strip():
                                chunks.append(f"[P{i+1}S{j//500+1}] {chunk}")
                    
                    if (i + 1) % 5 == 0:
                        gc.collect()
                        
                    # Limit total chunks to prevent too many API calls
                    if len(chunks) >= 50:  # Reduced from 100
                        logging.warning(f"Limiting chunks to 50 for API efficiency")
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
        """Generate embeddings in small batches with rate limiting"""
        if not self.chunks:
            return
        
        logging.info(f"Generating embeddings for {len(self.chunks)} chunks...")
        try:
            # Process in very small batches with rate limiting
            batch_size = 5  # Much smaller batches
            all_embeddings = []
            
            for i in range(0, len(self.chunks), batch_size):
                batch = self.chunks[i:i+batch_size]
                
                # Apply rate limiting before each API call
                rate_limiter.wait_if_needed()
                
                try:
                    result = genai.embed_content(
                        model='models/text-embedding-004',  # Use latest embedding model
                        content=batch,
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                    
                    if isinstance(result['embedding'][0], list):
                        all_embeddings.extend(result['embedding'])
                    else:
                        all_embeddings.append(result['embedding'])
                    
                    logging.info(f"Processed embedding batch {i//batch_size + 1}/{(len(self.chunks) + batch_size - 1)//batch_size}")
                    
                except Exception as e:
                    logging.error(f"Error in embedding batch {i//batch_size + 1}: {e}")
                    # Continue with next batch instead of failing completely
                    continue
                
                gc.collect()
            
            if all_embeddings:
                self.embeddings = np.array(all_embeddings, dtype=np.float32)
                logging.info("Embeddings generated successfully")
            else:
                logging.error("No embeddings were generated")
                self.embeddings = None
                
        except Exception as e:
            logging.error(f"Failed to generate embeddings: {e}")
            self.embeddings = None

    def get_relevant_context(self, query: str, top_k: int = 3) -> str:
        if self.embeddings is None:
            return "Error: Document embeddings not available."
        
        try:
            # Apply rate limiting before query embedding
            rate_limiter.wait_if_needed()
            
            result = genai.embed_content(
                model='models/text-embedding-004',
                content=query,
                task_type="RETRIEVAL_QUERY"
            )
            query_embedding = np.array(result['embedding'], dtype=np.float32)
            
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
        
        prompt = f"""Based ONLY on this policy context, answer the question precisely:

CONTEXT:
{context[:3000]}  # Increased context length for better answers

QUESTION: {question}

Answer with exact policy details (amounts, periods, conditions). If not in context, say "Not specified in policy document."

ANSWER:"""
        
        try:
            # Apply rate limiting before generation
            rate_limiter.wait_if_needed()
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=400,
                    top_p=0.9,
                )
            )
            return response.text.strip()
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            if "429" in str(e) or "quota" in str(e).lower():
                return "Error: API rate limit exceeded. Please try again later."
            return f"Error: {str(e)[:100]}"

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

    # Limit questions to prevent rate limit issues
    if len(questions) > 10:  # Reduced from 15
        questions = questions[:10]
        logging.warning("Limited to first 10 questions for API efficiency")

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        processor = MemoryEfficientDocumentProcessor()
        if not processor.download_file(doc_url, tmp_path):
            return jsonify({'error': f'Failed to download document from URL: {doc_url}'}), 500
        
        logging.info(f"Download completed. Memory: {get_memory_usage():.1f}MB")
        
        document_chunks = processor.process_pdf_streaming(tmp_path)
        if not document_chunks:
            return jsonify({'error': 'Failed to extract text from document.'}), 500

        logging.info(f"Processing completed. Memory: {get_memory_usage():.1f}MB")

        retriever = MemoryEfficientRetriever(document_chunks)
        if retriever.embeddings is None:
            return jsonify({'error': 'Failed to generate document embeddings.'}), 500
            
        logging.info(f"Embeddings completed. Memory: {get_memory_usage():.1f}MB")

        answer_engine = OptimizedAnsweringEngine(GEMINI_MODEL)
        
        # Process questions sequentially to avoid rate limits
        answers = []
        for i, question in enumerate(questions):
            try:
                logging.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
                context = retriever.get_relevant_context(question)
                answer = answer_engine.get_answer(question, context)
                answers.append(answer)
                
                # Add small delay between questions
                if i < len(questions) - 1:  # Don't sleep after last question
                    time.sleep(1)  # 1 second delay between questions
                    
            except Exception as e:
                logging.error(f"Error processing question {i+1}: {e}")
                answers.append(f"Error processing question: {str(e)[:50]}")
            
            gc.collect()
        
        total_time = time.time() - start_time
        final_memory = get_memory_usage()
        
        logging.info(f"All questions processed in {total_time:.2f}s. Memory: {final_memory:.1f}MB")
        
        return jsonify({
            'answers': answers,
            'processing_time': f"{total_time:.2f}s",
            'memory_used': f"{final_memory:.1f}MB",
            'questions_processed': len(answers)
        })

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.error(traceback.format_exc())
        return jsonify({'error': f'Internal server error: {str(e)[:100]}'}), 500
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
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
