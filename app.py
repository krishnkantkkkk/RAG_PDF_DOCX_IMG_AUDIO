import hashlib
import os
from flask import Flask, Response, render_template, request, jsonify, stream_with_context
import pickle
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from faster_whisper import WhisperModel
import tempfile
from flask import json
import docx
import math

app = Flask(__name__, static_folder='temp', static_url_path='/temp')

# Global variables
pdf_documents = []   # all documents from all files (PDFs, DOCX, Audio)
vector_store = None
loaded_hashes = set()  # keep track of already loaded file hashes
llm = ChatOllama(model="gemma3:4b")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")

CACHE_DIR = "cache"

# Helper to process a single PDF
def process_pdf(file):
    pdf_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(pdf_path)

    doc = fitz.open(pdf_path)
    page_data = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        images = []

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            ext = base_image["ext"]
            img_filename = f"{file.filename}_page{page_num+1}_img{img_index}.{ext}"
            img_path = os.path.join("temp", img_filename)
            with open(img_path, "wb") as f_img:
                f_img.write(img_bytes)
            images.append(img_path)

        page_data.append({
            "text": text,
            "images": images,
            "page_num": page_num,
            "pdf_filename": file.filename,
            "type": "pdf"
        })
    doc.close()
    return page_data

# Helper to process a DOCX file
def process_docx(file):
    docx_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(docx_path)

    doc = docx.Document(docx_path)
    full_text = "\n".join([para.text for para in doc.paragraphs])
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(full_text)

    doc_data = []
    for i, chunk in enumerate(chunks):
        doc_data.append({
            "text": chunk,
            "images": [],
            "page_num": i,
            "pdf_filename": file.filename,
            "type": "docx"
        })
    return doc_data

# 👇 NEW: Helper to process an Audio file
def process_audio(file):
    audio_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(audio_path)

    segments, _ = whisper_model.transcribe(audio_path, beam_size=5)
    
    # Group segments into larger chunks for better context
    chunk_data = []
    current_chunk_text = ""
    start_time = 0
    chunk_index = 0
    
    for i, segment in enumerate(segments):
        if not current_chunk_text:
            start_time = segment.start
            
        current_chunk_text += segment.text + " "
        
        # Chunk by length or after a certain number of segments
        if len(current_chunk_text) > 800 or (i > 0 and i % 10 == 0):
            chunk_data.append({
                "text": current_chunk_text.strip(),
                "images": [],
                "page_num": chunk_index,
                "pdf_filename": file.filename,
                "start_time": start_time,
                "end_time": segment.end,
                "type": "audio"
            })
            current_chunk_text = ""
            chunk_index += 1
    
    # Add the last remaining chunk
    if current_chunk_text:
        chunk_data.append({
            "text": current_chunk_text.strip(),
            "images": [],
            "page_num": chunk_index,
            "pdf_filename": file.filename,
            "start_time": start_time,
            "end_time": segment.end,
            "type": "audio"
        })
        
    return chunk_data

# Helper to create vector store for a set of documents
def create_vector_store(documents):
    embeddings = embedding_model.encode([doc['text'] for doc in documents])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Retriever that searches across all loaded files
def get_retriever():
    def retriever_fn(query, k=2):
        query_emb = embedding_model.encode([query])
        distances, indices = vector_store.search(query_emb, k)
        return [pdf_documents[i] for i in indices[0]]
    return retriever_fn

@app.route('/')
def index():
    return render_template('index.html')

# 👇 MODIFIED: The upload function now handles PDF, DOCX, and Audio
@app.route('/upload', methods=['POST'])
def upload_file():
    global pdf_documents, vector_store, loaded_hashes

    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        os.makedirs(CACHE_DIR, exist_ok=True)
        uploaded_files = []

        for file in files:
            file_bytes = file.read()
            file.seek(0)
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            cache_path = os.path.join(CACHE_DIR, file_hash)
            os.makedirs(cache_path, exist_ok=True)

            if file_hash in loaded_hashes:
                uploaded_files.append(file.filename)
                continue

            if os.path.exists(os.path.join(cache_path, "documents.pkl")) and os.path.exists(os.path.join(cache_path, "vector_store.faiss")):
                with open(os.path.join(cache_path, "documents.pkl"), "rb") as f:
                    docs = pickle.load(f)
                index = faiss.read_index(os.path.join(cache_path, "vector_store.faiss"))
            else:
                filename = file.filename.lower()
                if filename.endswith('.pdf'):
                    docs = process_pdf(file)
                elif filename.endswith('.docx'):
                    docs = process_docx(file)
                # --- 👇 NEW: Logic to select the audio processor ---
                elif filename.endswith(('.mp3', '.wav', '.m4a', '.ogg')):
                    docs = process_audio(file)
                else:
                    continue # Skip unsupported files

                index = create_vector_store(docs)
                with open(os.path.join(cache_path, "documents.pkl"), "wb") as f:
                    pickle.dump(docs, f)
                faiss.write_index(index, os.path.join(cache_path, "vector_store.faiss"))

            pdf_documents.extend(docs)

            if vector_store is None:
                vector_store = index
            else:
                new_embeddings = embedding_model.encode([doc['text'] for doc in docs])
                vector_store.add(new_embeddings)

            loaded_hashes.add(file_hash)
            uploaded_files.append(file.filename)

        return jsonify({
            'message': 'Files processed successfully',
            'filenames': uploaded_files
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    global vector_store, pdf_documents
    if vector_store is None:
        return jsonify({'error': 'No documents uploaded yet'}), 400

    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    retriever = get_retriever()
    retrieved_docs = retriever(question)

    context_text = "\n\n".join([doc['text'] for doc in retrieved_docs])

    template = """Answer the question based only on the following context and answer should be in html format enclosed within <div> tag. Do not use markdown formatting or styles:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = prompt | llm | StrOutputParser()

    def generate():
        for chunk in rag_chain.stream({"context": context_text, "question": question}):
            yield chunk
            
        # 👇 MODIFIED: Source object now includes timestamp info if available
        sources = []
        for doc in retrieved_docs:
            source_obj = {
                "pdf_filename": doc['pdf_filename'],
                "page_num": doc.get('page_num', 0) + 1,
                "source_content": doc['text'],
                "type": doc.get('type', 'unknown')
            }
            if 'start_time' in doc:
                source_obj['start_time'] = doc['start_time']
                source_obj['end_time'] = doc['end_time']
            sources.append(source_obj)

        yield json.dumps({"type": "sources", "content": sources})

    return Response(stream_with_context(generate()), mimetype='text/plain')


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        try:
            segments, _ = whisper_model.transcribe(temp_audio_path, beam_size=5)
            transcription = " ".join([segment.text for segment in segments])
        finally:
            os.remove(temp_audio_path)
        return jsonify({'transcription': transcription})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/files', methods=['GET'])
def list_files():
    filenames = list({doc['pdf_filename'] for doc in pdf_documents})
    return jsonify({'files': filenames})

if __name__ == '__main__':
    app.run(debug=True)
