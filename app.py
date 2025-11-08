from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import ToolMessage
from langchain.agents.middleware import wrap_tool_call, ModelCallLimitMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from dotenv import load_dotenv
import torch
import gradio as gr
import random
import re
import requests
from langsmith import traceable
import os
import json
from guardrails import Guard
from guardrails.hub import ToxicLanguage, ValidLength
from helpers.gradio_helpers import upload_button_handler, delete_button_click
from chroma_config import client

load_dotenv()

deepl_api_key = os.environ.get("DEEPL_API_KEY")

RESUME_K = 6      
EXAMPLES_K = 3     
MAX_RETRIES = 1 

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
COLLECTION_NAME = "cv_knowledge_base"

def _get_collection():
    return client.get_or_create_collection(COLLECTION_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChatGoogleGenerativeAI(
    temperature=0.6,
    model="gemini-2.0-flash",
    timeout=120
)
tokenizer = AutoTokenizer.from_pretrained('Vamsi/T5_Paraphrase_Paws')
human_model = AutoModelForSeq2SeqLM.from_pretrained('Vamsi/T5_Paraphrase_Paws').to(device)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

input_guard = Guard().use_many(
    ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail="filter"), # type: ignore
    ValidLength(min=40, max = 5000, on_fail="exception") # type: ignore
)
vectorstore = _get_collection()

def add_human_touch(text:str) -> str:
    """Add subtle human-like variations."""
    if random.random() > 0.4:
        text = re.sub(r'\bI am\b', "I'm", text, count=random.randint(1, 2))
    
    text = text.replace("I am particularly drawn to", "What excites me most about")
    text = text.replace("aligns perfectly with", "really resonates with")
    
    text = text.replace("Additionally,", "Also,")
    text = text.replace("Furthermore,", "Plus,")
 
    text = text.replace("Thank you for considering my application", 
                       "Thanks for taking the time to review my application")
    
    return text

@wrap_tool_call
def handle_tool_error(request, handler):
    """handle tool call errors gracefully."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(content=f"An error occurred while executing the tool: {str(e)}")


@tool
@traceable(run_type="tool")
def humanize_text(text: str) -> str:
    """ Run this model below Once, to produce the humanized result of your response. """
    
    print("Enhancing text...")
    paragraphs = text.split('\n\n')
    humanized_paragraph = []
    
    for para in paragraphs:
        if not para.strip():
            continue
        
        par = "paraphrase: " + para.strip() + " </s>"
        encoding = tokenizer.encode_plus(par, return_tensors='pt', max_length=512, truncation=True)
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
        
        
        outputs = human_model.generate(input_ids=input_ids, attention_mask=attention_masks, max_length = 512,top_k=120,
        top_p=0.95,do_sample=True,
        early_stopping=True)
        
        humanized = tokenizer.decode(outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
        humanized_paragraph.append(humanized)
        print(f"Processed paragraph: {humanized[:100]}...")
    
    result = '\n\n'.join(humanized_paragraph)
    final_result = add_human_touch(result)
    guarded_res = input_guard.validate(final_result)
    return guarded_res # type: ignore
    

@tool
@traceable(run_type="chain")    
def translate_to_ch(text : str) -> str:
    """ Make request to this api, this api translates the english text into chinese."""
    url = "https://api-free.deepl.com/v2/translate"
    headers = {
        "Authorization": f"DeepL-Auth-Key {deepl_api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "text" : [text],
        "target_lang" : "ZH-HANT"
    }
    
    response = requests.post(url, headers=headers, data= json.dumps(data))
    
    if response.status_code == 200:
        result = response.json()
        translated_text = result["translations"][0]["text"]
        print(translated_text[100:])
        return translated_text
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return "Translation failed"

agent = create_agent(
    model=model, 
    tools=[humanize_text, translate_to_ch], 
    middleware=[
        ModelCallLimitMiddleware(thread_limit=10, exit_behavior="end"),  # type: ignore
        handle_tool_error
    ]
) 

cv_results = 10
example_result = 6
@traceable(run_type="llm")
def stream_response(company_profile, job_desc):
        if isinstance(company_profile, list):
            company_profile = company_profile[0] if company_profile else ""
        if isinstance(job_desc, list):
            job_desc = job_desc[0] if job_desc else ""
        
        company_profile = str(company_profile)
        job_desc = str(job_desc)

        print(job_desc)
        input_guard.validate(company_profile.strip())
        input_guard.validate(job_desc.strip())
        
        example_results = vectorstore.query( # type: ignore
            query_texts = [company_profile + " " + job_desc], 
            n_results=example_result,
            where={"doc_type" : "samples"}
            )
        
        all_cv_results = vectorstore.query( # type: ignore
            query_texts=[job_desc],
            n_results=cv_results * 5,  
            where={"doc_type": "user_cv"}
        )
        
        class Document:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata
        
        example_docs = []
        
        if example_results['documents']:
            for i, content in enumerate(example_results['documents'][0]): # type: ignore
                metadata = example_results['metadatas'][0][i] if example_results['metadatas'] else {} # type: ignore
                example_docs.append(Document(page_content=content, metadata=metadata))

        all_cv_docs = []
        if all_cv_results['documents']:
            for i, content in enumerate(all_cv_results['documents'][0]): # type: ignore
                metadata = all_cv_results['metadatas'][0][i] if all_cv_results['metadatas'] else {} # type: ignore
                all_cv_docs.append(Document(page_content=content, metadata=metadata))

        print(f"Found {len(example_docs)} example documents from CLOUD")
        print(f"Found {len(all_cv_docs)} CV documents from CLOUD")
        
        def filter_docs_by_section(docs, section_name):
            print(docs)
            return [doc for doc in docs if doc.metadata.get("section") == section_name]
        
        summary_docs = filter_docs_by_section(all_cv_docs, "summary")[:cv_results]
        projects_docs = filter_docs_by_section(all_cv_docs, "projects")[:cv_results]
        experience_docs = filter_docs_by_section(all_cv_docs, "experience")[:cv_results]
        education_docs = filter_docs_by_section(all_cv_docs, "education")[:cv_results]
        certificate_docs = filter_docs_by_section(all_cv_docs, "certificates")[:cv_results]
        skills_docs = filter_docs_by_section(all_cv_docs, "skills")[:cv_results]
        
        
        example_knowledge = "\n\n".join([doc.page_content for doc in example_docs])
        cv_knowledge = f"""
        SUMMARY:
        {chr(10).join([doc.page_content for doc in summary_docs]) if summary_docs else "No projects found"}
        PROJECTS:
        {chr(10).join([doc.page_content for doc in projects_docs]) if projects_docs else "No projects found"}
        EXPERIENCE:
        {chr(10).join([doc.page_content for doc in experience_docs]) if experience_docs else "No experience found"}
        EDUCATION:
        {chr(10).join([doc.page_content for doc in education_docs]) if education_docs else "No education found"}
        CERTIFICATES:
        {chr(10).join([doc.page_content for doc in certificate_docs]) if certificate_docs else "No certificates found"}
        SKILLS:
        {chr(10).join([doc.page_content for doc in skills_docs]) if skills_docs else "No skills found"}
        """
        
            
        rag_prompt = f""" You are an AI assistant that helps users craft personalized cover letters for job applications. 
        Do NOT use any internal knowledge outside the provided sections.
        Do not invent facts. 
        Do not use "-" in the cover letter. 
        Be specific about technologies and project names.
        Aim to generate between 300-315 words in the letter.
        
        company_profile (user input): 
        {company_profile}
        
        Job description (user input): 
        {job_desc}
        
        User's CV Details (Use these SPECIFIC details in your cover letter):
        {cv_knowledge}
        
        COVER LETTER EXAMPLES (for reference on style and structure ONLY):
        {example_knowledge}
        
        Task: 
        1. CAREFULLY read through the USER'S CV section to identify:
       - Specific projects the user has worked on
       - Technologies and tools they've used
       - Quantifiable achievements and results
       - Relevant experience that matches the job requirements
    
        2. Match the job requirements with the user's ACTUAL experience from their CV:
       - Look for direct technology matches
       - Look for skill category matches
       - Use the strongest, most relevant examples from the CV
       - Be specific about technologies and project names
    
        3. Write a cover letter that:
       - Uses SPECIFIC projects and achievements from the user's CV
       - Mentions actual technologies the user has worked with
       - References real results and outcomes from their experience
       - Connects their experience to the job requirements
    
        4. Use the cover letter examples ONLY for:
       - Structure and professional tone
       - Effective phrasing and transitions
       - DO NOT copy content from examples
       - DO NOT use generic statements from examples
        
        5. Ensure the cover letter passes ATS scanning by including relevant keywords from the job description that match the user's actual experience
    
        CRITICAL: You MUST reference specific projects, technologies, and achievements from the USER'S CV section. Do not write a generic cover letter.
        
        Important: After you generate your final response, run the humanize_text tool, and after the humanize_text return the new response, run the translate_to_ch tool and give the 
        output back as your final result.
        """
        
        result = agent.invoke({
            "messages":[HumanMessage(content=rag_prompt)]})
        
        draft = result['messages'][-1].content
        input_guard.validate(draft)
        yield draft
    
with gr.Blocks() as app:
    state = gr.State({})
    
    gr.Markdown("# AI Cover Letter Generator")
    
    company_profile = gr.Textbox(label="Company Profile", lines=10, placeholder="Paste the company profile here...")
    job_desc = gr.Textbox(label="Job Description", lines=10, placeholder="Paste the job description here...")
    
    
    cv_file = gr.File(label="Upload CV (PDF)", file_count = "single", type="filepath") # type: ignore
    
    status = gr.Textbox(label="Status", interactive=False)
    
    upload_btn = gr.Button("Upload")
    delete_btn = gr.Button("Delete")
    
    output_box = gr.Textbox(label="Generated Cover Letter", lines=20, show_copy_button=True)
    
    upload_btn.click(upload_button_handler, inputs=[cv_file, state], outputs=[status, state])
    delete_btn.click(delete_button_click, inputs=[state], outputs=[status, state])
    
    generate_btn = gr.Button("Generate")
    
    generate_btn.click(fn=stream_response, inputs=[company_profile, job_desc], outputs=[output_box])
    
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1",
        server_port=7860
    )
    

