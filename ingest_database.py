from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4


# import the .env file
#from dotenv import load_dotenv
#load_dotenv()


# configuration
DATA_PATH = r"/Users/harinikolluru/Chatbot-with-RAG-and-LangChain/data"
CHROMA_PATH = r"chroma_db"


# initiate the embeddings model
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# initiate the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)


# loading the PDF document
loader = PyPDFDirectoryLoader(DATA_PATH)


raw_documents = loader.load()


print(f"\n[DEBUG] Loaded {len(raw_documents)} documents from: {DATA_PATH}")
for i, doc in enumerate(raw_documents[:3]):
    print(f"[DEBUG] Document {i+1} sample:\n", doc.page_content[:500])  # show first 500 characters




# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)


# creating the chunks
chunks = text_splitter.split_documents(raw_documents)


print(f"\n[DEBUG] Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks[:3]):
    print(f"[DEBUG] Chunk {i+1} sample:\n", chunk.page_content[:300])  # show chunk content




# creating unique ID's
uuids = [str(uuid4()) for _ in range(len(chunks))]


# adding chunks to vector store
vector_store.add_documents(documents=chunks, ids=uuids)


print("\n[DEBUG] Added documents to Chroma vector store.")
