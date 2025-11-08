import gradio as gr
import uuid
import os
from file_handler.upload_cv import upload_cv, delete_user_cv 

def get_or_create_user_id(state):
    if not state.get("user_id"):
        state["user_id"] = str(uuid.uuid4())
    return state["user_id"]

def resolve_uploaded_file(uploaded):
    if uploaded is None:
        return None
    
    if hasattr(uploaded, "name") and os.path.exists(uploaded.name):
        return uploaded
    if isinstance(uploaded, str) and os.path.exists(uploaded):
        class F: pass
        f = F(); f.name = uploaded # type: ignore
        return f
    raise ValueError("Can't resolve uploaded file object. Received: " + str(type(uploaded)))

def upload_button_handler(uploaded_file, state):
    user_id = get_or_create_user_id(state)
    if not uploaded_file:
        return "No file uploaded.", state
    
    try:
        fobj  =resolve_uploaded_file(uploaded_file)
        chunk_ids = upload_cv(fobj, user_id=user_id, replace  =True) # type: ignore
        return f"Uploaded {os.path.basename(fobj.name)} ({len(chunk_ids)} chunks) — session {user_id}", state # type: ignore
    except Exception as e:
        return f"Upload failed: {e}", state
    

def delete_button_click(state):
    user_id = get_or_create_user_id(state)
    try:
        delete_user_cv(user_id= user_id)
        return f"Deleted CVs for session {user_id}", state
    except Exception as e:
        return f"Delete failed: {e}", state
    

    
        
        