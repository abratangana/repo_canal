from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()


collection_name = 'Biblia'
qdrant_client = QdrantClient(url="http://localhost:6333")

model_embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=model_embeddings,
    retrieval_mode=RetrievalMode.DENSE,
    content_payload_key="text",
)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 7,
        "score_threshold": 0.5
    }
)

# ==== DEFINICIÓN DEL MODELO DE ENTRADA ====

class QueryInput(BaseModel):
    query: str

# ==== ENDPOINT PARA USO DE N8N ====

@app.post("/search_biblia")
def search_biblia(input: QueryInput):
    try:
        docs = retriever.get_relevant_documents(input.query)
        results = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

## ====Comando para ejecutar el servidor FastAPI====
#==== uvicorn busqueda:app --host 0.0.0.0 --port 8000 --reload ====

# ==== ENDPOINT PARA VER EN NAVEGADOR CON GET ====

@app.get("/buscar")
def buscar(query: str):
    docs = retriever.invoke(query)
    return docs

# ==== Ej: http://localhost:8000/buscar?query=paciencia%20en%20el%20libro%20de%20los%20romanos  ====    