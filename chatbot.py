from langchain_community.llms import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import gradio as gr


# import the .env file
#from dotenv import load_dotenv
#load_dotenv()


# configuration
DATA_PATH = r"/Users/harinikolluru/Chatbot-with-RAG-and-LangChain/data"
CHROMA_PATH = r"chroma_db"


embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# initiate the model
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
    temperature=0.5,
)


# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)


# Set up the vectorstore to be the retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})


def get_context(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context


# call this function for every message added to the chatbot
def stream_response(message, history):


    context = get_context(message)


    print("\n[DEBUG] Retrieved Context:\n", context)


    rag_prompt = f"""
You are a strict assistant that only answers questions based on the provided State Farm awards context.

## Context
{context}

## Question
{message}

INSTRUCTIONS (follow strictly, do not output):
- If the question doesn’t clearly mention awards, say: "I am a State Farm award managing assistant, and I can only answer any award related question you have. What can I help you with?"
- If the answer is not directly in the context, say: "I am a State Farm award managing assistant, and I can only answer any award related question you have. What can I help you with?"
- If asked who received the most (aliases, teams, etc.), only answer if complete data is available. If tied, list all tied entries with bullet points. 

## Answer 

"""


    partial_message = ""
    for response in llm.stream(rag_prompt):
        partial_message += response
        yield partial_message






# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)


# launch the Gradio app
chatbot.launch(share=True)
