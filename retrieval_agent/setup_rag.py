from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Step 1: Load documents
docs = []
doc_dir = "pubmed/"
for file in os.listdir(doc_dir):
    with open(os.path.join(doc_dir, file), 'r') as f:
        docs.append(f.read())

# Step 2: Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.create_documents(docs)

# Step 3: Embed & Save
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)
db.save_local("faiss_index/")