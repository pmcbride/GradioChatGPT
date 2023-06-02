#%%
import os
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, load_tools
import gradio as gr 

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


# search = GoogleSearchAPIWrapper()
# tools = [
#     Tool(
#     name ="Search" ,
#     func=search.run,
#     description="useful when you need to answer questions about current events"
#     ),
# ]
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

#%%
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_chain = initialize_agent(tools=tools,
                               llm=llm,
                               agent="chat-conversational-react-description",
                               verbose=True,
                               memory=memory)

def chat_response(input_text):
    response = agent_chain.run(input=input_text)
    return response

interface = gr.Interface(fn=chat_response, inputs="text", outputs="text", description="Chat with a conversational agent")

interface.launch(debug=True, share=True, server_name="0.0.0.0")