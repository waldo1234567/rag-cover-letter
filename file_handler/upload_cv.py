
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
from datetime import datetime
import chromadb
from chroma_config import client
import re
from typing import Dict

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "cv_knowledge_base"

CV_SECTIONS = {
    "projects": r"^(PROJECTS?|PORTFOLIO)",
    "skills": r"^(SKILLS?|TECHNICAL SKILLS?|COMPETENCIES)",
    "education": r"^(EDUCATION|ACADEMIC BACKGROUND)",
    "certificates": r"^(CERTIFICATES?|CERTIFICATIONS?)",
    "experience": r"^(EXPERIENCE|EMPLOYMENT|WORK HISTORY|WORK EXPERIENCE|OTHER EXPERIENCES)",
    "summary": r"^(SUMMARY|PROFILE|ABOUT|OBJECTIVE)"
}


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def splic_cv_by_section(full_text : str) -> Dict[str, str]:
    sections ={}
    current_section = "summary"
    current_content = []
    
    lines = full_text.split('\n')
    
    for line in lines:
        line_upper = line.strip().upper()
        
        found_section = None
        for section_name, pattern in CV_SECTIONS.items():
            if re.match(pattern, line_upper):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                current_section = section_name
                current_content = [line]  
                found_section = True
                break
        if not found_section:
            current_content.append(line)
            
    if current_content:
        sections[current_section] = '\n'.join(current_content)
    return sections

def _get_collection():
    return client.get_or_create_collection(COLLECTION_NAME)

def upload_cv(pdf_file, user_id:str, replace: bool = True, namespace: str | None = None):
    
    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()
    
    full_text = "\n\n".join([doc.page_content for doc in documents])
    
    sections = splic_cv_by_section(full_text=full_text)        
        
    text_splitters =  RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 300,
        length_function = len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitters.split_documents(documents)
    filename = os.path.basename(pdf_file.name)
    filename_noext = os.path.splitext(filename)[0]
    
    enhanced_chunks = []
    chunk_ids = []
    documents_list = [] 
    metadatas_list = []
    
    chunk_counter = 0
    section_counts = {}
    
    for section_name, section_content in sections.items():
        if not section_content.strip():
            continue
        
        section_doc = Document(
            page_content=section_content,
            metadata={"section": section_name}
        )
        
        section_chunks = text_splitters.split_documents([section_doc])
        
        section_counts[section_name] = len(section_chunks)
        
        for chunk_idx, chunk in enumerate(section_chunks):
            chunk_id = f"user_{user_id}_{filename_noext}_chunk{chunk_counter}"
            
            metadata = {
                "id": chunk_id,
                "source": filename,
                "doc_type": "user_cv",
                "section": section_name,  
                "title": filename_noext[:80],
                "chunk_index": chunk_counter,
                "section_chunk_index": chunk_idx, 
                "user_id": str(user_id),
                "uploaded_at": datetime.utcnow().isoformat() + "Z"
            }
            
            chunk.metadata.update(metadata)
            enhanced_chunks.append(chunk)
            chunk_ids.append(chunk_id)
            
            documents_list.append(chunk.page_content)
            metadatas_list.append(metadata)
            
            chunk_counter += 1
    

    
    shown_sections = set()
    for chunk in enhanced_chunks:
        section = chunk.metadata.get('section', 'unknown')
        if section not in shown_sections:
            shown_sections.add(section)
            if len(shown_sections) >= 4:  # Show max 4 samples
                break
        
    collection = _get_collection()
    
    if replace:
        try:
            if filename:
                where = {"$and": [
                    {"user_id": {"$eq": str(user_id)}},
                    {"source": {"$eq": filename}}
                ]}
            else:
                where = {"user_id": {"$eq": str(user_id)}}
            collection.delete(where=where) # type: ignore
        except Exception as e:
            try:
                delete_user_cv(user_id = user_id, filename=filename, namespace = namespace)
            except Exception as e:
                print(str(e))
                pass
    
    try:
        collection.add(ids = chunk_ids, documents=documents_list, metadatas=metadatas_list)
        print(f"Added {len(enhanced_chunks)} chunks for user {user_id} from {filename}")
        return chunk_ids
        
    except Exception as e:
        print(f"Error adding documents: {e}")
        raise


def delete_user_cv(user_id: str, filename: str | None = None, namespace: str | None = None):
    filter_dict = {"user_id" : str(user_id)}
    collection = _get_collection()
    if filename:
        filter_dict["source"] = filename
        
    try:
        collection.delete(where=filter_dict) # type: ignore
        print(f"Deleted documents for user {user_id}" + (f" from {filename}" if filename else ""))
        return
    except TypeError:
        try:
            collection.delete(where=filter_dict) # type: ignore
            print(f"Deleted documents for user {user_id}" + (f" from {filename}" if filename else ""))
            return
        except Exception:
            pass
    except Exception:
        pass
    
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        try:
            coll = client.get_collection(name=COLLECTION_NAME)
        except Exception as e:
            if "does not exist" in str(e).lower():
                print(f"Collection {COLLECTION_NAME} doesn't exist - nothing to delete")
                return
            else:
                raise
        if filename:
            where = {"user_id": str(user_id), "source": filename}
        else:
            where = {"user_id": str(user_id)}
    
        coll.delete(where=where) # type: ignore
        print(f"(chroma fallback) Deleted documents for user {user_id}")
        return
    except Exception as e:
        if "does not exist" in str(e).lower():
            print(f"Collection {COLLECTION_NAME} doesn't exist - nothing to delete")
            return
        raise RuntimeError(
            "Could not delete user CV documents via vectorstore.delete or chromadb client. "
            "Check your Chroma/LangChain versions and APIs. "
            "Error: " + repr(e)
        ) from e
    
def tag_samples():
    vectorstore= _get_collection()
    
    all_docs = vectorstore.get(include=["metadatas", "documents"])
    
    for i,metadata in enumerate(all_docs["metadatas"]): # type: ignore
        if not metadata.get("user_id"):
            vectorstore.update(
                ids=[all_docs["ids"][i]],
                metadatas=[{**metadata, "is_sample" : True}]
            )
            
    print("samples tagged")
    