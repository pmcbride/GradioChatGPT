#%% Imports
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool, load_tools
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings

# General 
import os
import pandas as pd
from langchain.experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain.chat_models import ChatOpenAI

from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.docstore.document import Document
import asyncio
import nest_asyncio


# Needed synce jupyter runs an async eventloop
nest_asyncio.apply()

import faiss
import argparse

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


# parser = argparse.ArgumentParser()
# # take command line arguments
# parser.add_argument("--goals", type=str, default=None)
# parser.add_argument("--tools", type=str, default=None)
# parser.add_argument("--ai_name", type=str, default=None)

llm = ChatOpenAI(temperature=0)
# llm = ChatOpenAI(model_name="gpt-4", temperature=1.0)

#%% Tools
import os
from contextlib import contextmanager
from typing import Optional
from langchain.agents import tool
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool

ROOT_DIR = "./data/"

@contextmanager
def pushd(new_dir):
    """Context manager for changing the current working directory."""
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)

@tool
def process_csv(
    csv_file_path: str, instructions: str, output_path: Optional[str] = None
) -> str:
    """Process a CSV by with pandas in a limited REPL.\
 Only use this after writing data to disk as a csv file.\
 Any figures must be saved to disk to be viewed by the human.\
 Instructions should be written in natural language, not code. Assume the dataframe is already loaded."""
    with pushd(ROOT_DIR):
        try:
            df = pd.read_csv(csv_file_path)
        except Exception as e:
            return f"Error: {e}"
        agent = create_pandas_dataframe_agent(llm, df, max_iterations=30, verbose=True)
        if output_path is not None:
            instructions += f" Save output to disk at {output_path}"
        try:
            result = agent.run(instructions)
            return result
        except Exception as e:
            return f"Error: {e}"

async def async_load_playwright(url: str) -> str:
    """Load the specified URLs using Playwright and parse using BeautifulSoup."""
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright

    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)

            page_source = await page.content()
            soup = BeautifulSoup(page_source, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            results = "\n".join(chunk for chunk in chunks if chunk)
        except Exception as e:
            results = f"Error: {e}"
        await browser.close()
    return results

def run_async(coro):
    event_loop = asyncio.get_event_loop()
    return event_loop.run_until_complete(coro)

@tool
def browse_web_page(url: str) -> str:
    """Verbose way to scrape a whole webpage. Likely to cause issues parsing."""
    return run_async(async_load_playwright(url))

#%% Q&A Over a webpage
from langchain.tools import BaseTool, DuckDuckGoSearchRun, GoogleSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pydantic import Field
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain, BaseCombineDocumentsChain

def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
    )


class WebpageQATool(BaseTool):
    name = "query_webpage"
    description = "Browse a webpage and retrieve the information relevant to the question."
    text_splitter: RecursiveCharacterTextSplitter = Field(default_factory=_get_text_splitter)
    qa_chain: BaseCombineDocumentsChain
    
    def _run(self, url: str, question: str) -> str:
        """Useful for browsing websites and scraping the text information."""
        result = browse_web_page.run(url)
        docs = [Document(page_content=result, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)
        results = []
        # TODO: Handle this with a MapReduceChain
        for i in range(0, len(web_docs), 4):
            input_docs = web_docs[i:i+4]
            window_result = self.qa_chain({"input_documents": input_docs, "question": question}, return_only_outputs=True)
            results.append(f"Response from window {i} - {window_result}")
        results_docs = [Document(page_content="\n".join(results), metadata={"source": url})]
        return self.qa_chain({"input_documents": results_docs, "question": question}, return_only_outputs=True)
    
    async def _arun(self, url: str, question: str) -> str:
        raise NotImplementedError
      

#%% Memory
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools.human.tool import HumanInputRun

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

#%% Tool List
# web_search = GoogleSearchRun(data={"max_results": 10})
web_search = load_tools(["google-search"], llm=llm)[0]
query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))
serpapi_tool = load_tools(["serpapi"], llm=llm)[0]

tools = [
    web_search,
    # serpapi_tool,
    WriteFileTool(root_dir="./data"),
    ReadFileTool(root_dir="./data"),
    # process_csv,
    query_website_tool,
    HumanInputRun(), # Activate if you want the permit asking for help from the human
]

#%% AutGPT
agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever(search_kwargs={"k": 8}),
    # human_in_the_loop=True, # Set to True if you want to add feedback at each step.
)
# agent.chain.verbose = True

goals = [
    "Write a python ChatGPT clone that uses gradio as the front end and a conversational langchain OpenAI gpt-4 chat model as the backend.",
    "The chatbot should be able to search the internet, read and answer questions about files and websites, and store all previously looked up information in a faiss memory vectorstore (langchain.vectorstores.FAISS) using OpenAI embeddings (langchain.embeddings.OpenAIEmbeddings) for quick QA retrieval.",
    "Start by crawling through the gradio documentation at `https://gradio.app/docs` and the langchain documentation at `https://python.langchain.com/en/latest/` and embedding the information in the memory vectorstore.",
    "Remember to write a python file that uses gradio to run the chatbot and displays the Human and AI responses using gradio's gr.Chatbot class.",
]
response = agent.run(goals)
print(response)
# %%
