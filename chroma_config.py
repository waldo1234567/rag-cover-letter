import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

chromadb_api_key = os.environ.get("CHROMA_API_KEY")
COLLECTION_NAME = "cv_knowledge_base"

client = chromadb.CloudClient(
        api_key= chromadb_api_key,
        tenant='cf995d4e-5a38-4917-9b31-0fc2f01b0cff',
        database=COLLECTION_NAME)