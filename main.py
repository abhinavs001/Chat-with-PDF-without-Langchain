import json
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index.core import ServiceContext, set_global_service_context
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.llms.gradient import GradientBaseModelLLM
from copy import deepcopy
from tempfile import NamedTemporaryFile
from pydantic import BaseModel
# all the libraries are imported
#create app for the FastAPI generation
app = FastAPI()

class Document(BaseModel):
    content: bytes
    filename: str
#connect to the vector database that is Apache Cassandra
cloud_config = {
    'secure_connect_bundle': 'secure-connect-checkdatabase.zip'
}

with open("token.json") as f:
    secrets = json.load(f)

CLIENT_ID = secrets["clientId"]
CLIENT_SECRET = secrets["secret"]

auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astra_session = cluster.connect()
#Set up Gradient for LLAMA2 and embeddings
os.environ['GRADIENT_ACCESS_TOKEN'] = ''
os.environ['GRADIENT_WORKSPACE_ID'] = ''

llm = GradientBaseModelLLM(
    base_model_slug="llama2-7b-chat",
    max_tokens=400,
)

embed_model = GradientEmbedding(
    gradient_access_token=os.environ["GRADIENT_ACCESS_TOKEN"],
    gradient_workspace_id=os.environ["GRADIENT_WORKSPACE_ID"],
    gradient_model_slug="bge-large",
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    chunk_size=256,
)

set_global_service_context(service_context)

@app.post("/upload/")
async def upload_file(doc: UploadFile = File(...)):
    with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
        f.write(doc.file.read())
        documents = SimpleDirectoryReader(".").load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        query_engine = index.as_query_engine()
        return {"query_engine": query_engine}

@app.post("/query/")
async def query_text(text: str):
    return {"response": "Response from processing the text"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
#uvicorn.run(app, host="127.0.0.1", port=5000)