import os
import gradio as gr 
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, load_tools
from langchain.utilities import SerpAPIWrapper, GoogleSearchAPIWrapper
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
import faiss

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


search = SerpAPIWrapper()
tools = [
    Tool(
        name = "search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    WriteFileTool(),
    ReadFileTool(),
]
# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    memory=vectorstore.as_retriever()
)
# Set verbose to be true
agent.chain.verbose = True

def chat_response(input_text):
    response = agent.run(input=input_text)
    return response

interface = gr.Interface(fn=chat_response, inputs="text", outputs="text", description="Chat with a conversational agent")

interface.launch(debug=True, share=True, server_name="0.0.0.0")