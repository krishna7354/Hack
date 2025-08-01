import os
import traceback
import logging
import gc
import time
import json
import hashlib
import re
import asyncio
import aiohttp
import aiofiles
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
from datetime import datetime, timedelta

# --- Configuration ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
HACKRX_API_KEY = os.getenv('HACKRX_API_KEY')

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required!")

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GeminiAPIManager:
    """Optimized Gemini API Manager for high concurrency with free tier limits"""
    
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        # Use Gemini Flash for free tier - much higher rate limits
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
        
        # Reduced sessions for free tier
        self.session_pool = []
        for _ in range(3):  # Only 3 sessions for free tier
            session = requests.Session()
            session.headers.update({
                'Content-Type': 'application/json'
            })
            self.session_pool.append(session)
        
        self.session_index = 0
        self.session_lock = threading.Lock()
        
        # Conservative rate limiting for free tier
        self.last_request_times = [0] * len(self.session_pool)
        self.rate_limit_delay = 2.0  # 2 seconds between requests per session for free tier
    
    def get_session(self) -> Tuple[requests.Session, int]:
        """Get next available session in round-robin fashion"""
        with self.session_lock:
            session_idx = self.session_index % len(self.session_pool)
            self.session_index += 1
            return self.session_pool[session_idx], session_idx
    
    def _wait_for_rate_limit(self, session_idx: int):
        """Per-session rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_times[session_idx]
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_times[session_idx] = time.time()

    def _clean_final_answer(self, text: str) -> str:
        """Clean up the final model output"""
        # Remove markdown and formatting
        text = re.sub(r'\*\*', '', text)
        text = re.sub(r'^\s*[-*]\s*', '', text, flags=re.MULTILINE)
        
        lines = text.split('\n')
        # Get the first substantial line
        for line in lines:
            line = line.strip()
            if line and len(line) > 10 and not line.endswith(':'):
                return line
        
        return text.strip()

    def get_optimized_answer(self, question: str, context: str) -> Tuple[str, str]:
        """Get answer using Gemini 1.5 Pro"""
        
        session, session_idx = self.get_session()
        
        # Optimized prompt for Gemini
        prompt = f"""You are an insurance policy expert. Based ONLY on the provided policy document context, answer the question with a single, direct sentence.

RULES:
- Answer in ONE sentence only (maximum 30 words)
- Use ONLY information from the context
- Be precise and factual
- If information is not in context, say "Information not found in the provided document"

Context:
{self._optimize_context(context, question)}

Question: {question}

Answer:"""

        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 60,
                "stopSequences": ["\n\n"]
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
        }
        
        try:
            self._wait_for_rate_limit(session_idx)
            
            url = f"{self.base_url}?key={self.api_key}"
            response = session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0]['content']['parts'][0]['text']
                answer = self._clean_final_answer(content)
                
                if len(answer) < 10 or len(answer) > 200:
                    raise Exception("Response length out of bounds")
                
                return answer, "gemini-1.5-pro"
            else:
                raise Exception("No valid response from Gemini")
                
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            raise

    def _optimize_context(self, context: str, question: str) -> str:
        """Optimize context for the specific question"""
        chunks = context.split('\n\n---\n\n')
        
        question_lower = question.lower()
        question_keywords = set(re.findall(r'\b\w{3,}\b', question_lower))
        
        # Insurance-specific keywords
        insurance_keywords = [
            'premium', 'grace period', 'waiting period', 'coverage', 'benefit', 'claim',
            'policy', 'insured', 'hospital', 'treatment', 'medical', 'maternity',
            'pre-existing', 'disease', 'surgery', 'discount', 'limit', 'sub-limit',
            'ayush', 'room rent', 'icu', 'organ donor', 'health check', 'ncd'
        ]
        
        # Score chunks
        scored_chunks = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            chunk_words = set(re.findall(r'\b\w{3,}\b', chunk_lower))
            
            # Score based on keyword matches
            base_score = len(question_keywords.intersection(chunk_words))
            insurance_score = sum(1 for keyword in insurance_keywords if keyword in chunk_lower)
            
            # Bonus for exact phrase matches
            phrase_bonus = sum(2 for word in question_keywords if len(word) > 4 and word in chunk_lower)
            
            # Bonus for numerical information
            number_bonus = len(re.findall(r'\d+\s*(?:days?|months?|years?|%|\$|₹)', chunk_lower))
            
            total_score = base_score + insurance_score + phrase_bonus + number_bonus
            scored_chunks.append((total_score, chunk))
        
        # Get top 6 most relevant chunks
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        relevant_chunks = [chunk for score, chunk in scored_chunks[:6] if score > 0]
        
        if not relevant_chunks:
            relevant_chunks = [chunk for score, chunk in scored_chunks[:4]]
        
        optimized_context = '\n\n--- POLICY SECTION ---\n\n'.join(relevant_chunks)
        
        # Ensure reasonable length for Gemini
        if len(optimized_context) > 12000:
            optimized_context = optimized_context[:12000]
            last_period = optimized_context.rfind('.')
            if last_period > 8000:
                optimized_context = optimized_context[:last_period + 1]
        
        return optimized_context

    def process_multiple_questions(self, questions: List[str], context: str) -> List[Dict]:
        """Process multiple questions with high concurrency"""
        results = []
        
        # Conservative workers for free tier
        max_workers = min(len(questions), 2)  # Only 2 concurrent requests for free tier
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all questions
            future_to_question = {
                executor.submit(self._process_single_question, q, context, i): q 
                for i, q in enumerate(questions)
            }
            
            # Collect results
            for future in as_completed(future_to_question):
                question = future_to_question[future]
                try:
                    result = future.result(timeout=30)
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
        
        # Sort by original order
        results.sort(key=lambda x: questions.index(x['question']) if x['question'] in questions else 999)
        return results

    def _process_single_question(self, question: str, context: str, index: int) -> Dict:
        """Process a single question with timing"""
        start_time = time.time()
        
        try:
            answer, model_used = self.get_optimized_answer(question, context)
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
                'model_used': 'gemini-1.5-pro',
                'processing_time': round(processing_time, 2),
                'status': 'failed'
            }


class FastDocumentProcessor:
    """Ultra-fast document processor with aggressive caching"""
    
    def __init__(self):
        self.chunk_cache = {}
        self.processed_urls = {}  # Cache by URL
        self.cache_lock = threading.Lock()
    
    def get_url_hash(self, url: str) -> str:
        """Generate hash for URL caching"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def download_file(self, url: str, target_path: str) -> bool:
        """Fast download with retries"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/pdf,*/*'
            }
            
            # Use stream=False for small PDFs to reduce overhead
            response = requests.get(url, timeout=20, headers=headers, stream=False)
            response.raise_for_status()
            
            with open(target_path, 'wb') as f:
                f.write(response.content)
            
            logging.info(f"Downloaded file: {len(response.content)} bytes")
            return True
            
        except Exception as e:
            logging.error(f"Download failed: {e}")
            return False
    
    def get_cached_chunks(self, url: str) -> Optional[List[str]]:
        """Get cached chunks for URL"""
        with self.cache_lock:
            url_hash = self.get_url_hash(url)
            if url_hash in self.processed_urls:
                cache_time, chunks = self.processed_urls[url_hash]
                # Cache valid for 1 hour
                if time.time() - cache_time < 3600:
                    logging.info("Using cached document chunks")
                    return chunks
        return None
    
    def cache_chunks(self, url: str, chunks: List[str]):
        """Cache chunks for URL"""
        with self.cache_lock:
            url_hash = self.get_url_hash(url)
            self.processed_urls[url_hash] = (time.time(), chunks)
    
    def process_pdf_fast(self, file_path: str, url: str) -> List[str]:
        """Ultra-fast PDF processing"""
        
        # Check URL cache first
        cached_chunks = self.get_cached_chunks(url)
        if cached_chunks:
            return cached_chunks
        
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                # Process fewer pages for speed, but smarter selection
                pages_to_process = min(20, total_pages)
                
                logging.info(f"Fast processing {pages_to_process} pages from PDF")
                
                for page_num in range(pages_to_process):
                    try:
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        
                        if text and len(text.strip()) > 30:
                            cleaned_text = self._fast_clean_text(text)
                            page_chunks = self._fast_chunk(cleaned_text, page_num + 1)
                            chunks.extend(page_chunks)
                        
                        # Early stopping if we have enough content
                        if len(chunks) >= 40:
                            break
                            
                    except Exception as e:
                        logging.warning(f"Error processing page {page_num + 1}: {e}")
                        continue
                
                # Cache the results
                self.cache_chunks(url, chunks)
                logging.info(f"Fast processed {len(chunks)} chunks from PDF")
                
        except Exception as e:
            logging.error(f"PDF processing failed: {e}")
            return []
        
        return chunks
    
    def _fast_clean_text(self, text: str) -> str:
        """Fast text cleaning"""
        # Basic cleaning only
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        return text.strip()
    
    def _fast_chunk(self, text: str, page_num: int) -> List[str]:
        """Fast chunking strategy"""
        chunks = []
        
        # Simple sentence-based chunking
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = []
        current_length = 0
        target_length = 300  # Smaller chunks for faster processing
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence.split())
            
            if current_chunk and current_length + sentence_length > target_length:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) > 100:
                    chunks.append(f"[Page {page_num}] {chunk_text}")
                
                # Start new chunk with minimal overlap
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) > 100:
                chunks.append(f"[Page {page_num}] {chunk_text}")
        
        return chunks


class FastRetriever:
    """Lightning-fast context retrieval"""
    
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.keyword_index = self._build_fast_index()
    
    def _build_fast_index(self) -> Dict[str, List[int]]:
        """Build minimal keyword index"""
        index = {}
        
        for i, chunk in enumerate(self.chunks):
            words = re.findall(r'\b\w{4,}\b', chunk.lower())  # Only 4+ char words
            
            for word in set(words[:20]):  # Limit words per chunk
                if word not in index:
                    index[word] = []
                index[word].append(i)
        
        return index
    
    def get_relevant_context(self, question: str, top_k: int = 8) -> str:
        """Fast context retrieval"""
        question_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
        
        chunk_scores = {}
        
        for word in question_words:
            if word in self.keyword_index:
                for chunk_idx in self.keyword_index[word]:
                    chunk_scores[chunk_idx] = chunk_scores.get(chunk_idx, 0) + 1
        
        if chunk_scores:
            sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
            top_chunks = [self.chunks[idx] for idx, score in sorted_chunks[:top_k]]
        else:
            top_chunks = self.chunks[:top_k]
        
        return '\n\n---\n\n'.join(top_chunks)


# Initialize global components
gemini_manager = GeminiAPIManager()
doc_processor = FastDocumentProcessor()

# --- API Endpoints ---
@app.before_request
def check_authentication():
    """Authentication middleware"""
    if request.endpoint == 'run_hackrx':
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authorization header with Bearer token required'}), 401
        
        try:
            api_key = auth_header.split(' ')[1]
            if HACKRX_API_KEY and api_key != HACKRX_API_KEY:
                return jsonify({'error': 'Invalid API key'}), 403
        except IndexError:
            return jsonify({'error': 'Invalid Authorization header format'}), 401

@app.route('/hackrx/run', methods=['POST'])
def run_hackrx():
    """High-performance HackRX API endpoint"""
    start_time = time.time()
    tmp_path = None
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body must be valid JSON'}), 400
        
        if 'documents' not in data or 'questions' not in data:
            return jsonify({'error': 'Missing required fields: documents, questions'}), 400
        
        doc_url = data['documents']
        questions = data['questions']
        
        if not isinstance(doc_url, str) or not doc_url.strip():
            return jsonify({'error': 'documents must be a non-empty string URL'}), 400
        
        if not isinstance(questions, list) or len(questions) == 0:
            return jsonify({'error': 'questions must be a non-empty list'}), 400
        
        for i, question in enumerate(questions):
            if not isinstance(question, str) or not question.strip():
                return jsonify({'error': f'Question {i+1} must be a non-empty string'}), 400
        
        logging.info(f"Processing {len(questions)} questions with high-performance pipeline")
        
        # Fast document processing
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_path = tmp.name
        tmp.close()
        
        if not doc_processor.download_file(doc_url, tmp_path):
            return jsonify({'error': 'Failed to download document'}), 500
        
        chunks = doc_processor.process_pdf_fast(tmp_path, doc_url)
        
        if not chunks:
            return jsonify({'error': 'Could not extract content from PDF'}), 500
        
        logging.info(f"Extracted {len(chunks)} chunks in {time.time() - start_time:.2f}s")
        
        # Fast retrieval and processing
        retriever = FastRetriever(chunks)
        all_questions_text = ' '.join(questions)
        unified_context = retriever.get_relevant_context(all_questions_text, top_k=10)
        
        # High-concurrency question processing
        results = gemini_manager.process_multiple_questions(questions, unified_context)
        
        # Extract answers in order
        answers = []
        for question in questions:
            for result in results:
                if result['question'] == question:
                    if result['status'] == 'success':
                        answers.append(result['answer'])
                    else:
                        answers.append("Unable to generate answer for this question.")
                    break
            else:
                answers.append("Answer not available.")
        
        processing_time = time.time() - start_time
        successful_answers = sum(1 for r in results if r['status'] == 'success')
        
        logging.info(f"Completed {successful_answers}/{len(questions)} in {processing_time:.2f}s")
        
        return jsonify({"answers": answers})
        
    except Exception as e:
        logging.error(f"Request failed: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        gc.collect()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'api': 'gemini-1.5-pro',
        'cache_size': len(doc_processor.processed_urls),
        'timestamp': time.time()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logging.info(f"Starting optimized Gemini Document Q&A API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
