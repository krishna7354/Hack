import os
import traceback
import logging
import gc
import time
import json
import hashlib
import re
from typing import List, Dict, Optional, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import PyPDF2
import requests
import tempfile
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
HACKRX_API_KEY = os.getenv('HACKRX_API_KEY')

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required!")

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GroqAPIManager:
    """Optimized Groq API Manager for maximum speed and accuracy"""
    
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.models = {
            'primary': 'llama-3.1-70b-versatile',      # Best accuracy
            'fast': 'llama-3.1-8b-instant',           # Fastest
            'backup': 'mixtral-8x7b-32768'             # Good balance
        }
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # CHANGED: Slower rate to prevent "Too Many Requests"
        self.request_lock = threading.Lock()
    
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last)
            self.last_request_time = time.time()

    def _clean_final_answer(self, text: str) -> str:
        """A simple function to clean up the final model output."""
        # Remove markdown bolding
        text = text.replace('**', '')
        
        lines = text.split('\n')
        # Filter out lines that look like headers or are empty
        good_lines = [line for line in lines if line.strip() and not line.strip().endswith(':')]
        
        # CHANGED: Join only the first line for a very short answer
        return ' '.join(good_lines[:1]).strip()

    def get_optimized_answer(self, question: str, context: str, use_fast_model: bool = False) -> Tuple[str, str]:
        """Get answer with optimized prompting to match expected response quality"""
        
        model = self.models['fast'] if use_fast_model else self.models['primary']
        
        # CHANGED: Made prompt even more restrictive for a single, short sentence.
        system_prompt = """You are an insurance policy expert. Your task is to provide an extremely concise answer based ONLY on the provided policy document context.

RESPONSE REQUIREMENTS:
1.  **Answer in ONE single, direct sentence ONLY.**
2.  **Keep the answer under 30 words.**
3.  **ABSOLUTELY NO MARKDOWN, headers, lists, or bullet points.**
4.  **DO NOT add any extra explanations or introductory phrases.**
5.  If the information is not in the text, respond with: "Information not found in the provided document."

EXAMPLE: "A grace period of thirty days is provided for premium payment."
"""

        user_prompt = f"""Policy Document Context:
{self._optimize_context(context, question)}

Question: {question}

Based on the context, provide one single, direct sentence:"""

        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            'max_tokens': 60,       # CHANGED: Reduced from 150 to 60
            'temperature': 0.0,
            'top_p': 1,
            'frequency_penalty': 0,
            'presence_penalty': 0,
            'stream': False
        }
        
        try:
            self._wait_for_rate_limit()
            
            response = self.session.post(
                self.base_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            raw_answer = result['choices'][0]['message']['content'].strip()
            
            answer = self._clean_final_answer(raw_answer)
            
            if len(answer) < 10 or ("not found" not in answer.lower() and len(answer) > 200):
                 raise Exception("Response is too long or too short after cleaning.")

            return answer, model
            
        except Exception as e:
            logging.error(f"Groq API error with {model}: {e}")
            if not use_fast_model and model == self.models['primary']:
                logging.info("Falling back to the fast model...")
                return self.get_optimized_answer(question, context, use_fast_model=True)
            raise
    
    def _optimize_context(self, context: str, question: str) -> str:
        """Optimize context for insurance policy questions"""
        # Split context into chunks
        chunks = context.split('\n\n---\n\n')
        
        # Extract key terms from question for better matching
        question_lower = question.lower()
        
        # Insurance-specific keywords that indicate important sections
        insurance_keywords = [
            'premium', 'grace period', 'waiting period', 'coverage', 'benefit', 'claim',
            'policy', 'insured', 'hospital', 'treatment', 'medical', 'maternity',
            'pre-existing', 'disease', 'surgery', 'discount', 'limit', 'sub-limit',
            'ayush', 'room rent', 'icu', 'organ donor', 'health check', 'ncd'
        ]
        
        # Score chunks based on question keywords and insurance terms
        question_keywords = set(re.findall(r'\b\w{3,}\b', question_lower))
        scored_chunks = []
        
        for chunk in chunks:
            chunk_lower = chunk.lower()
            chunk_words = set(re.findall(r'\b\w{3,}\b', chunk_lower))
            
            # Base score from question keyword matches
            base_score = len(question_keywords.intersection(chunk_words))
            
            # Bonus for insurance-specific terms
            insurance_score = sum(1 for keyword in insurance_keywords if keyword in chunk_lower)
            
            # Extra bonus for exact phrase matches
            phrase_bonus = 0
            for word in question_keywords:
                if len(word) > 4 and word in chunk_lower:
                    phrase_bonus += 2
            
            # Bonus for numerical information (amounts, percentages, time periods)
            number_bonus = len(re.findall(r'\d+\s*(?:days?|months?|years?|%|\$|₹)', chunk_lower))
            
            total_score = base_score + insurance_score + phrase_bonus + number_bonus
            scored_chunks.append((total_score, chunk))
        
        # Sort by relevance and take top chunks
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        
        # Take more chunks for complex questions
        num_chunks = 8 if any(word in question_lower for word in ['explain', 'describe', 'what are', 'how does']) else 6
        relevant_chunks = [chunk for score, chunk in scored_chunks[:num_chunks] if score > 0]
        
        # If no good matches, take first few chunks
        if not relevant_chunks:
            relevant_chunks = [chunk for score, chunk in scored_chunks[:4]]
        
        # Combine chunks with clear separation
        optimized_context = '\n\n--- POLICY SECTION ---\n\n'.join(relevant_chunks)
        
        # Ensure we don't exceed context limits while keeping important info
        if len(optimized_context) > 15000:
            # Truncate but try to keep complete sentences
            optimized_context = optimized_context[:15000]
            last_period = optimized_context.rfind('.')
            if last_period > 10000:  # Only truncate at sentence if it's not too early
                optimized_context = optimized_context[:last_period + 1]
            optimized_context += "\n\n[Note: Additional policy content available but truncated for processing]"
        
        return optimized_context
    
    def process_multiple_questions(self, questions: List[str], context: str) -> List[Dict]:
        """Process multiple questions with parallel execution"""
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=1) as executor: # CHANGED: From 3 to 1 to serialize requests
            # Submit all questions
            future_to_question = {
                executor.submit(self._process_single_question, q, context, i): q 
                for i, q in enumerate(questions)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_question):
                question = future_to_question[future]
                try:
                    result = future.result(timeout=25)  # 25s timeout per question
                    results.append(result)
                except Exception as e:
                    logging.error(f"Question failed: {question[:50]}... Error: {e}")
                    results.append({
                        'question': question,
                        'answer': f"Processing failed: {str(e)}",
                        'model_used': 'none',
                        'processing_time': 0,
                        'status': 'failed'
                    })
        
        # Sort results by original question order
        results.sort(key=lambda x: questions.index(x['question']) if x['question'] in questions else 999)
        return results
    
    def _process_single_question(self, question: str, context: str, index: int) -> Dict:
        """Process a single question with timing"""
        start_time = time.time()
        
        try:
            # Use fast model for simple questions, primary for complex ones
            use_fast = self._is_simple_question(question)
            
            answer, model_used = self.get_optimized_answer(question, context, use_fast)
            processing_time = time.time() - start_time
            
            return {
                'question': question,
                'answer': answer,
                'model_used': model_used,
                'processing_time': round(processing_time, 2),
                'status': 'success'
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"Question {index} failed: {e}")
            
            return {
                'question': question,
                'answer': f"Unable to process this question: {str(e)}",
                'model_used': 'none',
                'processing_time': round(processing_time, 2),
                'status': 'failed'
            }
    
    def _is_simple_question(self, question: str) -> bool:
        """Determine if question is simple enough for fast model"""
        # Use primary model for all questions to ensure quality
        return False  # Always use best model

class OptimizedDocumentProcessor:
    """Highly optimized document processor for speed and accuracy"""
    
    def __init__(self):
        self.chunk_cache = {}
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash for file caching"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def download_file(self, url: str, target_path: str) -> bool:
        """Download file with optimizations"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, stream=True, timeout=30, headers=headers)
            response.raise_for_status()
            
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logging.info(f"Downloaded file: {os.path.getsize(target_path)} bytes")
            return True
            
        except Exception as e:
            logging.error(f"Download failed: {e}")
            return False
    
    def process_pdf_optimized(self, file_path: str) -> List[str]:
        """Process PDF with maximum efficiency and context preservation"""
        
        # Check cache first
        file_hash = self.get_file_hash(file_path)
        if file_hash in self.chunk_cache:
            logging.info("Using cached PDF chunks")
            return self.chunk_cache[file_hash]
        
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                # Smart page selection - focus on important pages
                pages_to_process = min(25, total_pages)  # Process up to 25 pages
                
                logging.info(f"Processing {pages_to_process} pages from PDF")
                
                for page_num in range(pages_to_process):
                    try:
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        
                        if text and len(text.strip()) > 50:
                            # Clean and structure text
                            cleaned_text = self._clean_text(text)
                            
                            # Smart chunking - preserve context
                            page_chunks = self._create_smart_chunks(cleaned_text, page_num + 1)
                            chunks.extend(page_chunks)
                            
                        # Stop if we have enough quality content
                        if len(chunks) >= 60:  # Optimal number for processing speed
                            break
                            
                    except Exception as e:
                        logging.warning(f"Error processing page {page_num + 1}: {e}")
                        continue
                
                # Cache the results
                self.chunk_cache[file_hash] = chunks
                logging.info(f"Processed {len(chunks)} optimized chunks from PDF")
                
        except Exception as e:
            logging.error(f"PDF processing failed: {e}")
            return []
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Advanced text cleaning for better context"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)  # Space between numbers and letters
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Space after punctuation
        
        # Remove unwanted characters but keep structure
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        
        return text.strip()
    
    def _create_smart_chunks(self, text: str, page_num: int) -> List[str]:
        """Create contextually aware chunks optimized for insurance policies"""
        chunks = []
        
        # Look for section headers and important policy terms
        section_patterns = [
            r'(?i)(clause|section|article|part)\s*\d+',
            r'(?i)(benefit|coverage|exclusion|condition|definition)',
            r'(?i)(premium|claim|policy|waiting period|grace period)',
            r'(?i)(maternity|pre-existing|surgery|treatment|hospital)'
        ]
        
        # Split by sentences but keep policy structure
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = []
        current_length = 0
        target_length = 400  # Slightly larger chunks for policy content
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence.split())
            
            # Check if this sentence starts a new section
            is_section_start = any(re.search(pattern, sentence) for pattern in section_patterns)
            
            # If this is a section start and we have content, finish current chunk
            if is_section_start and current_chunk and current_length > 150:
                chunk_text = ' '.join(current_chunk)
                chunks.append(f"[Page {page_num}, Policy Section {len(chunks) + 1}] {chunk_text}")
                current_chunk = [sentence]
                current_length = sentence_length
            # If adding this sentence would exceed target, save current chunk
            elif current_chunk and current_length + sentence_length > target_length:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) > 120:  # Minimum chunk size for policy content
                    chunks.append(f"[Page {page_num}, Policy Section {len(chunks) + 1}] {chunk_text}")
                
                # Start new chunk with overlap for context continuity
                if len(current_chunk) > 2:
                    # Keep last 2 sentences for context
                    current_chunk = current_chunk[-2:] + [sentence]
                    current_length = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) > 120:
                chunks.append(f"[Page {page_num}, Policy Section {len(chunks) + 1}] {chunk_text}")
        
        return chunks

class SmartRetriever:
    """Intelligent context retrieval without embeddings for speed"""
    
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.keyword_index = self._build_keyword_index()
    
    def _build_keyword_index(self) -> Dict[str, List[int]]:
        """Build keyword index for fast retrieval"""
        index = {}
        
        for i, chunk in enumerate(self.chunks):
            # Extract important keywords
            words = re.findall(r'\b\w{3,}\b', chunk.lower())  # Words with 3+ chars
            
            for word in set(words):  # Remove duplicates
                if word not in index:
                    index[word] = []
                index[word].append(i)
        
        return index
    
    def get_relevant_context(self, question: str, top_k: int = 6) -> str:
        """Get most relevant context using advanced keyword matching"""
        
        # Extract question keywords
        question_words = set(re.findall(r'\b\w{3,}\b', question.lower()))
        
        # Score chunks based on keyword matches
        chunk_scores = {}
        
        for word in question_words:
            if word in self.keyword_index:
                for chunk_idx in self.keyword_index[word]:
                    if chunk_idx not in chunk_scores:
                        chunk_scores[chunk_idx] = 0
                    
                    # Weight by word frequency and position
                    chunk_scores[chunk_idx] += 1
                    
                    # Bonus for exact phrase matches
                    if word in self.chunks[chunk_idx].lower():
                        chunk_scores[chunk_idx] += 0.5
        
        # Get top scoring chunks
        if not chunk_scores:
            # Fallback: return first few chunks
            top_chunks = self.chunks[:top_k]
        else:
            sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
            top_chunk_indices = [idx for idx, score in sorted_chunks[:top_k]]
            top_chunks = [self.chunks[i] for i in top_chunk_indices]
        
        return '\n\n---\n\n'.join(top_chunks)

# Initialize global components
groq_manager = GroqAPIManager()
doc_processor = OptimizedDocumentProcessor()

# --- API Endpoints ---
@app.before_request
def check_authentication():
    """Authentication middleware for /hackrx/run endpoint"""
    if request.endpoint == 'run_hackrx':
        # Check Content-Type
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        # Check Accept header (optional but good practice)
        accept_header = request.headers.get('Accept', '')
        if accept_header and 'application/json' not in accept_header:
            return jsonify({'error': 'Accept header must include application/json'}), 406
        
        # Check Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Authorization header is required'}), 401
        
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authorization header must start with "Bearer "'}), 401
        
        # Extract and validate API key
        try:
            api_key = auth_header.split(' ')[1]
            if not api_key:
                return jsonify({'error': 'API key is required'}), 401
            
            # If HACKRX_API_KEY is set, validate it
            if HACKRX_API_KEY and api_key != HACKRX_API_KEY:
                return jsonify({'error': 'Invalid API key'}), 403
                
        except IndexError:
            return jsonify({'error': 'Invalid Authorization header format'}), 401

@app.route('/hackrx/run', methods=['POST'])
def run_hackrx():
    """
    Main HackRX API endpoint - exactly matching specification
    
    Expected Request:
    POST /hackrx/run
    Content-Type: application/json
    Accept: application/json
    Authorization: Bearer <api_key>
    
    Body: {
        "documents": "pdf_url",
        "questions": ["question1", "question2", ...]
    }
    
    Response: {
        "answers": ["answer1", "answer2", ...]
    }
    """
    start_time = time.time()
    tmp_path = None
    
    try:
        # Validate request body
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body must be valid JSON'}), 400
        
        # Validate required fields
        if 'documents' not in data:
            return jsonify({'error': 'Missing required field: documents'}), 400
        
        if 'questions' not in data:
            return jsonify({'error': 'Missing required field: questions'}), 400
        
        doc_url = data['documents']
        questions = data['questions']
        
        # Validate data types
        if not isinstance(doc_url, str) or not doc_url.strip():
            return jsonify({'error': 'documents must be a non-empty string URL'}), 400
        
        if not isinstance(questions, list) or len(questions) == 0:
            return jsonify({'error': 'questions must be a non-empty list'}), 400
        
        # Validate each question
        for i, question in enumerate(questions):
            if not isinstance(question, str) or not question.strip():
                return jsonify({'error': f'Question {i+1} must be a non-empty string'}), 400
        
        # Log the processing start
        logging.info(f"Processing request with {len(questions)} questions")
        logging.info(f"Document URL: {doc_url[:100]}...")
        
        # Download and process document
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_path = tmp.name
        tmp.close()
        
        logging.info("Downloading document...")
        if not doc_processor.download_file(doc_url, tmp_path):
            return jsonify({'error': 'Failed to download document from provided URL'}), 500
        
        logging.info("Processing PDF content...")
        chunks = doc_processor.process_pdf_optimized(tmp_path)
        
        if not chunks:
            return jsonify({'error': 'Could not extract readable content from PDF'}), 500
        
        logging.info(f"Extracted {len(chunks)} content chunks from document")
        
        # Initialize retriever for context extraction
        retriever = SmartRetriever(chunks)
        
        # Get optimized context for all questions
        all_questions_text = ' '.join(questions)
        unified_context = retriever.get_relevant_context(all_questions_text, top_k=10)
        
        # Process all questions
        logging.info(f"Generating answers for {len(questions)} questions...")
        results = groq_manager.process_multiple_questions(questions, unified_context)
        
        # Extract just the answers in order for the response
        answers = []
        for question in questions:
            # Find the result for this question
            answer_found = False
            for result in results:
                if result['question'] == question:
                    if result['status'] == 'success':
                        answers.append(result['answer'])
                    else:
                        answers.append("Unable to generate answer for this question.")
                    answer_found = True
                    break
            
            if not answer_found:
                answers.append("Answer not available.")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log completion
        successful_answers = sum(1 for r in results if r['status'] == 'success')
        logging.info(f"Completed in {processing_time:.2f}s - {successful_answers}/{len(questions)} successful")
        
        # Return response in exact format specified
        return jsonify({
            "answers": answers
        })
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error: {e}")
        return jsonify({'error': 'Failed to download document - network error'}), 500
        
    except Exception as e:
        logging.error(f"Unexpected error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error occurred'}), 500
    
    finally:
        # Cleanup temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logging.info("Cleaned up temporary file")
            except Exception as e:
                logging.warning(f"Failed to cleanup temp file: {e}")
        
        # Force garbage collection
        gc.collect()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test Groq API
        test_payload = {
            'model': groq_manager.models['fast'],
            'messages': [{'role': 'user', 'content': 'Hello'}],
            'max_tokens': 10
        }
        
        response = groq_manager.session.post(
            groq_manager.base_url,
            json=test_payload,
            timeout=5
        )
        
        groq_status = 'healthy' if response.status_code == 200 else 'degraded'
        
    except Exception as e:
        groq_status = 'unhealthy'
        logging.error(f"Groq health check failed: {e}")
    
    return jsonify({
        'status': 'healthy' if groq_status == 'healthy' else 'degraded',
        'groq_api': groq_status,
        'models_available': list(groq_manager.models.values()),
        'cache_size': len(doc_processor.chunk_cache),
        'timestamp': time.time()
    })

@app.route('/api/test', methods=['POST'])
def test_endpoint():
    """Test endpoint to verify API functionality"""
    try:
        data = request.get_json() or {}
        test_question = data.get('question', 'What is this document about?')
        test_context = data.get('context', 'This is a test document containing sample information about insurance policies and coverage details.')
        
        start_time = time.time()
        answer, model_used = groq_manager.get_optimized_answer(test_question, test_context)
        processing_time = time.time() - start_time
        
        return jsonify({
            'test_question': test_question,
            'test_answer': answer,
            'model_used': model_used,
            'processing_time': f"{processing_time:.2f}s",
            'status': 'API is working correctly'
        })
        
    except Exception as e:
        logging.error(f"Test endpoint error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'API test failed'
        }), 500

# @app.route('/api/sample', methods=['POST'])
# def sample_request():
#     """Sample endpoint showing exact request format"""
#     sample_request = {
#         "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
#         "questions": [
#             "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
#             "What is the waiting period for pre-existing diseases (PED) to be covered?",
#             "Does this policy cover maternity expenses, and what are the conditions?"
#         ]
#     }
    
#     sample_response = {
#         "answers": [
#             "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
#             "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
#             "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period."
#         ]
#     }
    
#     return jsonify({
#         'endpoint': '/hackrx/run',
#         'method': 'POST',
#         'headers': {
#             'Content-Type': 'application/json',
#             'Accept': 'application/json',
#             'Authorization': 'Bearer <your_api_key>'
#         },
#         'sample_request': sample_request,
#         'sample_response': sample_response,
#         'note': 'Use the exact format shown above for best results'
#     })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logging.info(f"Starting Groq-optimized Document Q&A API on port {port}")
    logging.info(f"Available models: {list(groq_manager.models.values())}")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
