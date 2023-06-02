#%% Imports
import os
from typing import List, Iterator
import gradio as gr
import requests
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import elevenlabs
from elevenlabs import generate, play, save, stream
import subprocess
import pprint as pp
import config

# LangChain
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, load_tools

openai.api_key = config.OPENAI_API_KEY
elevenlabs.set_api_key(config.ELEVEN_API_KEY)

#%% Initialize Agent
tool_names = [
    "google-search",                          # google search tool
    "serpapi",                          # serpapi_tool
    "ddg-search",                          # duckduckgo search tool
    "requests_get", "requests_post",    # requests_tools
    # "python_repl",                      # python_repl_tool
    "llm-math",                         # math_tools
]

llm = ChatOpenAI(model_name="gpt-4", temperature=0)
# llm = ChatOpenAI(temperature=0)
tools = load_tools(tool_names, llm=llm)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_chain = initialize_agent(tools=tools,
                               llm=llm,
                               agent="chat-conversational-react-description",
                               verbose=True,
                               memory=memory)

#%% Chat Functions

def add_text(history: List[List[str]], text):
    history = history + [(text, None)]
    return history, ""

def add_file(history: List[List[str]], file):
    history = history + [((file.name,), None)]
    return history

def user(user_message, history: List[List[str]]):
    history += [[user_message, None]]
    return history, ""

def print_agent_chat_history(history: List[List[str]]):
    messages = agent_chain.memory.chat_memory.messages
    for m in messages:
        print(f"{str(m.type.upper())}:".ljust(7) + f"{m.content}")

def chat_response(history: List[List[str]]) -> str:
    """Generate chat_response from agent_chain and update history
        input_keys:  ['input', 'chat_history']
        output_keys: ['output']
    """
    global agent_chain
    user_message = history[-1][0]
    # response = agent_chain({"input": user_message, "chat_history": history})
    response = agent_chain.run(user_message)
    return response

def bot(history: List[List[str]]) -> List[List[str]]:
    """ Generate response from agent_chain and update bot history. """
    response = chat_response(history)
    history[-1][1] = response
    return history

# set a custom theme
theme = gr.themes.Default().set(body_background_fill="#000000")




with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=750)

    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])
    (
        txt.submit(add_text, [chatbot, txt], [chatbot, txt])
        .then(bot, chatbot, chatbot)
    )
    (
        btn.upload(add_file, [chatbot, btn], [chatbot])
        .then(bot, chatbot, chatbot)
    )
#%% Launch
demo.launch(debug=True, share=True, server_name="0.0.0.0")
