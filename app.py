import hashlib
import os
import pickle
import io
import fitz  # PyMuPDF
import faiss
import docx
import tempfile
from flask import Flask, Response, render_template, request, jsonify, stream_with_context, json
from sentence_transformers import SentenceTransformer, CrossEncoder # ✨ IMPROVEMENT: Added CrossEncoder for reranking
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from faster_whisper import WhisperModel
import pytesseract
from PIL import Image
import numpy as np # ✨ IMPROVEMENT: Added for numpy operations

# --- App Initialization ---
app = Flask(__name__, static_folder='temp', static_url_path='/temp')

# --- Global Variables & Model Loading ---
all_documents_metadata = [] # All documents currently in memory
vector_store = None
session_uploaded_files = set() # ✨ NEW: Track files uploaded in current session
session_file_hashes = {} # ✨ NEW: Map filename -> hash for current session
session_file_indices = {} # ✨ NEW: Track start/end indices for each file in the vector store
CACHE_DIR = "cache"

# ✨ IMPROVEMENT: Using a powerful text embedding model for better semantic search. CLIP is great for image/text, but a dedicated text model often works better for text-heavy Q&A.
# We'll handle images separately. This is a design choice for higher text accuracy.
# For a truly unified space, a larger multimodal model like 'clip-ViT-L-14' would be a step up.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
llm = ChatOllama(model="gemma3:4b") # Changed to a slightly faster model for query expansion
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")

# ✨ IMPROVEMENT: Load the reranker model
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# --- Processing Functions ---

def process_pdf(file_storage):
    """
    Processes a PDF file by extracting text chunks and images.
    Performs OCR on images to extract text.
    """
    # Use in-memory buffer instead of saving to disk first
    file_bytes = io.BytesIO(file_storage.read())
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    processed_data = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for page_num, page in enumerate(doc):
        # 1. Process regular text on the page
        text = page.get_text()
        if text.strip():
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                processed_data.append({
                    "text": chunk,
                    "page_num": page_num + 1,
                    "source_filename": file_storage.filename,
                    "type": "docx"
                })

        # 2. Process images on the page
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                # Save the image file to be served to the frontend
                os.makedirs("temp", exist_ok=True)
                ext = base_image["ext"]
                img_filename = f"{os.path.splitext(file_storage.filename)[0]}_p{page_num+1}_{img_index}.{ext}"
                img_path = os.path.join("temp", img_filename)
                pil_image.save(img_path)

                # Perform OCR to get text from the image
                ocr_text = pytesseract.image_to_string(pil_image)
                
                # ✨ IMPROVEMENT: Better context for images
                # TODO: For a future step, you could add a VLM captioning model here
                # image_caption = generate_image_caption(pil_image)
                image_caption = "This is an image." # Placeholder
                
                image_doc_text = f"Context from an image on page {page_num + 1}. Description: {image_caption}. Text found in image: '{ocr_text}'".strip()

                processed_data.append({
                    "text": image_doc_text, # This text is what gets embedded
                    "image_path": img_filename,
                    "page_num": page_num + 1,
                    "source_filename": file_storage.filename,
                    "type": "image"
                })
            except Exception as e:
                print(f"Warning: Could not process image {img_index} on page {page_num+1} of {file_storage.filename}. Error: {e}")

    doc.close()
    return processed_data


def process_docx(file_storage):
    """Processes a DOCX file by splitting its text content into chunks."""
    file_bytes = io.BytesIO(file_storage.read())
    doc = docx.Document(file_bytes)
    full_text = "\n".join([para.text for para in doc.paragraphs])
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(full_text)

    doc_data = []
    for i, chunk in enumerate(chunks):
        doc_data.append({
            "text": chunk,
            "page_num": i + 1, # Use chunk index as a pseudo page number
            "source_filename": file_storage.filename,
            "type": "docx"
        })
    return doc_data


def process_audio(file_storage):
    """Processes an audio file by transcribing it with timestamps and creating smart chunks."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_storage.filename.split('.')[-1]}") as temp_audio:
        file_storage.save(temp_audio.name)
        temp_audio_path = temp_audio.name

    try:
        segments, _ = whisper_model.transcribe(temp_audio_path, beam_size=5)
        # Convert segments to list with timing info
        timed_segments = []
        for seg in segments:
            timed_segments.append({
                "text": seg.text.strip(),
                "start": seg.start,
                "end": seg.end
            })
    finally:
        os.remove(temp_audio_path)

    if not timed_segments:
        return []

    # Smart chunking: Group segments into chunks while preserving timing
    chunk_data = []
    current_chunk = {
        "text": "",
        "start_time": None,
        "end_time": None,
        "segments": []
    }
    
    current_length = 0
    max_chunk_size = 1000
    overlap_size = 150
    
    for segment in timed_segments:
        segment_text = segment["text"]
        segment_length = len(segment_text)
        
        # If adding this segment would exceed chunk size and we have content, finalize current chunk
        if current_length + segment_length > max_chunk_size and current_chunk["text"]:
            # Finalize current chunk
            chunk_data.append({
                "text": current_chunk["text"].strip(),
                "source_filename": file_storage.filename,
                "type": "audio",
                "start_time": current_chunk["start_time"],
                "end_time": current_chunk["end_time"],
                "duration": current_chunk["end_time"] - current_chunk["start_time"],
                "page_num": len(chunk_data) + 1  # Use chunk index as page number
            })
            
            # Start new chunk with overlap from previous chunk
            overlap_text = ""
            overlap_length = 0
            # Take last few segments for overlap
            for prev_seg in reversed(current_chunk["segments"]):
                if overlap_length + len(prev_seg["text"]) <= overlap_size:
                    overlap_text = prev_seg["text"] + " " + overlap_text
                    overlap_length += len(prev_seg["text"])
                else:
                    break
            
            current_chunk = {
                "text": overlap_text + " " + segment_text if overlap_text else segment_text,
                "start_time": segment["start"],
                "end_time": segment["end"],
                "segments": [segment]
            }
            current_length = len(current_chunk["text"])
        else:
            # Add segment to current chunk
            if current_chunk["start_time"] is None:
                current_chunk["start_time"] = segment["start"]
            current_chunk["end_time"] = segment["end"]
            
            if current_chunk["text"]:
                current_chunk["text"] += " " + segment_text
            else:
                current_chunk["text"] = segment_text
                
            current_chunk["segments"].append(segment)
            current_length = len(current_chunk["text"])
    
    # Don't forget the last chunk
    if current_chunk["text"]:
        chunk_data.append({
            "text": current_chunk["text"].strip(),
            "source_filename": file_storage.filename,
            "type": "audio",
            "start_time": current_chunk["start_time"],
            "end_time": current_chunk["end_time"],
            "duration": current_chunk["end_time"] - current_chunk["start_time"],
            "page_num": len(chunk_data) + 1
        })
        
    return chunk_data

def create_vector_store_from_docs(documents):
    """Creates a FAISS vector store from a list of document chunks."""
    texts = [doc['text'] for doc in documents]
    embeddings = embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.cpu().numpy())
    return index
    

def format_timestamp(seconds):
    """Convert seconds to MM:SS format."""
    if seconds is None:
        return "00:00"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


# ✨ IMPROVEMENT: Function to expand the user's query for better retrieval
def expand_query(query):
    """Uses an LLM to generate 3 alternative queries."""
    template = """
    Based on the user's question, generate 3 additional, different, and more specific queries that are likely to find relevant documents in a vector database.
    Focus on rephrasing, using synonyms, and breaking down the question into sub-questions.
    Provide ONLY the queries, each on a new line. Do not number them or add any other text.
    
    Original Question: {question}
    
    Generated Queries:
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"question": query})
        expanded_queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
        # Always include the original query
        all_queries = [query] + expanded_queries
        print(f"Expanded queries: {all_queries}")
        return list(set(all_queries)) # Use set to remove duplicates
    except Exception as e:
        print(f"Query expansion failed: {e}")
        return [query] # Fallback to original query


def load_from_cache(file_hash):
    """Load processed document data and embeddings from cache if they exist."""
    cache_path = os.path.join(CACHE_DIR, file_hash)
    docs_path = os.path.join(cache_path, "documents.pkl")
    embeddings_path = os.path.join(cache_path, "embeddings.npy")
    faiss_path = os.path.join(cache_path, "faiss_index.bin")
    
    if os.path.exists(docs_path):
        try:
            # Load documents
            with open(docs_path, "rb") as f:
                docs = pickle.load(f)
            
            # Load embeddings if available
            embeddings = None
            if os.path.exists(embeddings_path):
                embeddings = np.load(embeddings_path)
            
            # Load FAISS index if available
            faiss_index = None
            if os.path.exists(faiss_path):
                faiss_index = faiss.read_index(faiss_path)
                
            return {
                "docs": docs,
                "embeddings": embeddings,
                "faiss_index": faiss_index
            }
        except Exception as e:
            print(f"Could not load cache for {file_hash}: {e}")
    return None


def save_to_cache(file_hash, docs, embeddings=None, faiss_index=None):
    """Save processed document data, embeddings, and FAISS index to cache."""
    cache_path = os.path.join(CACHE_DIR, file_hash)
    os.makedirs(cache_path, exist_ok=True)
    
    # Save documents
    with open(os.path.join(cache_path, "documents.pkl"), "wb") as f:
        pickle.dump(docs, f)
    
    # Save embeddings if provided
    if embeddings is not None:
        np.save(os.path.join(cache_path, "embeddings.npy"), embeddings)
    
    # Save FAISS index if provided
    if faiss_index is not None:
        faiss.write_index(faiss_index, os.path.join(cache_path, "faiss_index.bin"))


# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global all_documents_metadata, vector_store, session_uploaded_files, session_file_hashes, session_file_indices
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        newly_processed_docs = []
        processed_filenames = []

        for file in files:
            file_bytes = file.read()
            file.seek(0)
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            
            # ✨ NEW: Check if this exact file is already in current session
            if file.filename in session_uploaded_files and session_file_hashes.get(file.filename) == file_hash:
                print(f"File {file.filename} already uploaded in current session, skipping...")
                continue
            
            # ✨ IMPROVED: Try to load from cache first (including vectors)
            cached_data = load_from_cache(file_hash)
            
            if cached_data is not None and cached_data["docs"] is not None:
                print(f"Loading {file.filename} from cache...")
                docs = cached_data["docs"]
                cached_embeddings = cached_data["embeddings"]
                
                # Update filenames in cached docs to match current upload
                for doc in docs:
                    doc['source_filename'] = file.filename
                
                # If we have cached embeddings, use them directly
                if cached_embeddings is not None:
                    print(f"Using cached embeddings for {file.filename}")
                    new_embeddings = cached_embeddings
                else:
                    print(f"Creating embeddings for cached documents of {file.filename}")
                    new_texts = [doc['text'] for doc in docs]
                    new_embeddings = embedding_model.encode(new_texts, convert_to_tensor=True).cpu().numpy()
                    # Save the embeddings for future use
                    save_to_cache(file_hash, docs, new_embeddings)
                    
            else:
                # Process the file as it's not in cache
                print(f"Processing new file: {file.filename}")
                
                filename = file.filename.lower()
                if filename.endswith('.pdf'):
                    docs = process_pdf(file)
                elif filename.endswith('.docx'):
                    docs = process_docx(file)
                elif filename.endswith(('.mp3', '.wav', '.m4a', '.ogg')):
                    docs = process_audio(file)
                else:
                    continue
                
                if not docs: 
                    continue
                
                # Create embeddings for new documents
                print(f"Creating embeddings for new file: {file.filename}")
                new_texts = [doc['text'] for doc in docs]
                new_embeddings = embedding_model.encode(new_texts, convert_to_tensor=True).cpu().numpy()
                
                # Save to cache for future use (including embeddings)
                save_to_cache(file_hash, docs, new_embeddings)
            
            # ✨ IMPROVED: Track where this file's vectors start and end in the global index
            start_idx = len(all_documents_metadata)
            end_idx = start_idx + len(docs)
            session_file_indices[file.filename] = {
                "start": start_idx,
                "end": end_idx,
                "count": len(docs)
            }
            
            # Add to global vectors and metadata
            if vector_store is None:
                dimension = new_embeddings.shape[1]
                vector_store = faiss.IndexFlatL2(dimension)
            
            vector_store.add(new_embeddings)
            all_documents_metadata.extend(docs)
            
            session_uploaded_files.add(file.filename)
            session_file_hashes[file.filename] = file_hash
            processed_filenames.append(file.filename)
            
            print(f"Added {len(docs)} chunks from {file.filename} to session (indices {start_idx}-{end_idx-1})")

        if processed_filenames:
            print(f"Successfully processed files: {processed_filenames}")
            print(f"Total documents in session: {len(all_documents_metadata)}")
            print(f"Vector store size: {vector_store.ntotal if vector_store else 0}")

        return jsonify({'message': 'Files processed successfully', 'filenames': processed_filenames})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ✨ MODIFIED: Remove the initialize_from_cache function since we start with empty session
# def initialize_from_cache():
#     """
#     This function is removed. We now start with an empty session and only load
#     files when they are explicitly uploaded by the user.
#     """
#     pass


@app.route('/ask', methods=['POST'])
def ask_question():
    if vector_store is None or vector_store.ntotal == 0: 
        return jsonify({'error': 'No documents uploaded yet'}), 400
    
    data = request.get_json()
    question = data.get('question')
    if not question: return jsonify({'error': 'No question provided'}), 400

    # ✨ IMPROVEMENT: Step 1 - Expand the query
    queries = expand_query(question)
    query_embeddings = embedding_model.encode(queries, convert_to_tensor=True).cpu().numpy()

    # ✨ IMPROVEMENT: Step 2 - Retrieve a larger candidate set
    k_retrieval = 20 # Retrieve more documents initially for the reranker
    distances, ids = vector_store.search(query_embeddings, k_retrieval)
    
    unique_ids = set()
    for id_list in ids:
        for i in id_list:
            if i != -1:
                unique_ids.add(i)
    
    candidate_docs = [all_documents_metadata[i] for i in unique_ids]
    
    if not candidate_docs:
        return Response(stream_with_context(iter(["<div><p>I couldn't find any relevant information in the uploaded documents to answer your question.</p></div>"])))

    # ✨ IMPROVEMENT: Step 3 - Rerank the candidates
    rerank_pairs = [[question, doc['text']] for doc in candidate_docs]
    scores = reranker.predict(rerank_pairs)
    
    # Combine docs with their new scores
    doc_scores = list(zip(candidate_docs, scores))
    # Sort by the new score in descending order
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select the top 5 after reranking
    top_k_reranked = 2
    retrieved_results = doc_scores[:top_k_reranked]
    
    retrieved_docs_metadata = [res[0] for res in retrieved_results]
    
    # Build context for the LLM
    context_text = "\n\n".join([f"Source from {doc['source_filename']}, Page/Chunk {doc.get('page_num', 'N/A')}:\n{doc['text']}" for doc in retrieved_docs_metadata])
    
    template = "Answer the question based ONLY on the following context. Your answer must be in HTML format, enclosed within a single <div> tag. Do not use markdown, backticks, or any styling. Be concise and directly address the question.\n\nContext:\n{context}\n\nQuestion: {question}"
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = prompt | llm | StrOutputParser()

    def generate():
        # Stream the LLM's generated answer
        for chunk in rag_chain.stream({"context": context_text, "question": question}):
            yield chunk
            
        # After streaming the answer, send the source information
        sources = []
        for doc, score in retrieved_results:
            source_obj = {
                "source_filename": doc['source_filename'],
                "page_num": doc.get('page_num', 0),
                "source_content": doc['text'],
                "type": doc.get('type', 'unknown'),
                "score": float(score)  # Use the more meaningful reranker score
            }
            if doc.get('type') == 'image': 
                source_obj['image_path'] = doc['image_path']
            
            # ✨ FIXED: Add audio timing information
            if doc.get('type') == 'audio':
                source_obj['start_time'] = doc.get('start_time')
                source_obj['end_time'] = doc.get('end_time')
                source_obj['duration'] = doc.get('duration')
                # Format timestamps for display
                source_obj['timestamp_display'] = f"{format_timestamp(doc.get('start_time'))} - {format_timestamp(doc.get('end_time'))}"
            
            sources.append(source_obj)
        yield json.dumps({"type": "sources", "content": sources})

    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Endpoint to transcribe spoken audio from the frontend."""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        # Use a temporary file to save the audio blob from the browser
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        
        try:
            # Transcribe the audio file
            segments, _ = whisper_model.transcribe(temp_audio_path, beam_size=5)
            transcription = " ".join([segment.text for segment in segments])
        finally:
            # Clean up the temporary file
            os.remove(temp_audio_path)
            
        return jsonify({'transcription': transcription})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/files', methods=['GET'])
def list_files():
    """✨ MODIFIED: Returns detailed info about files uploaded in the current session."""
    file_info = []
    for filename in sorted(session_uploaded_files):
        info = {
            "filename": filename,
            "hash": session_file_hashes.get(filename),
            "indices": session_file_indices.get(filename, {}),
            "chunk_count": session_file_indices.get(filename, {}).get("count", 0)
        }
        file_info.append(info)
    
    return jsonify({
        'files': list(session_uploaded_files),
        'detailed_info': file_info,
        'total_chunks': len(all_documents_metadata),
        'vector_store_size': vector_store.ntotal if vector_store else 0
    })


# ✨ NEW: Route to get session statistics
@app.route('/session-info', methods=['GET'])
def session_info():
    """Get detailed information about the current session."""
    return jsonify({
        'uploaded_files': list(session_uploaded_files),
        'file_indices': session_file_indices,
        'total_documents': len(all_documents_metadata),
        'vector_store_size': vector_store.ntotal if vector_store else 0,
        'cache_stats': {
            'cached_files': len([d for d in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, d))]) if os.path.exists(CACHE_DIR) else 0
        }
    })


# ✨ NEW: Route to clear current session
@app.route('/clear-session', methods=['POST'])
def clear_session():
    """Clear all uploaded files from the current session."""
    global all_documents_metadata, vector_store, session_uploaded_files, session_file_hashes, session_file_indices
    
    all_documents_metadata = []
    vector_store = None
    session_uploaded_files.clear()
    session_file_hashes.clear()
    session_file_indices.clear()
    
    return jsonify({'message': 'Session cleared successfully'})


if __name__ == '__main__':
    # ✨ MODIFIED: Start with empty session instead of loading from cache
    print("Starting with empty session. Upload files to begin.")
    app.run(debug=True, port=5000)
