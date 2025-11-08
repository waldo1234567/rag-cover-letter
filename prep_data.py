from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from chroma_config import client
from langchain_huggingface import HuggingFaceEmbeddings
import os

from dotenv import load_dotenv
load_dotenv()

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
COLLECTION_NAME = "cv_base_knowledge"

def _get_collection():
    return client.get_or_create_collection(COLLECTION_NAME)

vectorstore = _get_collection()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False
)

chunks = text_splitter.split_documents(documents)
chunks_ids = []
enchaned_chunks = []



for doc in chunks:
    source = doc.metadata.get("source", "unknown_source")
    filename = os.path.splitext(os.path.basename(source))[0]
    chunk_idx = doc.metadata.get("chunk", None)
    if chunk_idx is None:
        cid = f"{filename}_{abs(hash(doc.page_content)) % 10_000_000}"
    else:
        cid = f"{filename}_chunk{chunk_idx}"
        
    doc_type = "cv" if "cv" in filename.lower() else "samples" if "samples" in filename.lower() else "other"

    doc.metadata.update({
        "id": cid,
        "source": source,
        "doc_type": doc_type,
        "title": filename[:80]    
    })
    chunks_ids.append(chunk_idx)
    enchaned_chunks.append(doc)
    
vectorstore.add(ids = chunks_ids, documents=enchaned_chunks)

print(f"Added {len(enchaned_chunks)} documents to the vector store.")


