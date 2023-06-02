from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.utilities import SerpAPIWrapper, GoogleSearchAPIWrapper
from langchain.tools.google_search import GoogleSearch

# Create a ChatOpenAI object with the desired parameters
chat = ChatOpenAI(temperature=0)

# Create a GoogleSearch object with the desired parameters
search = GoogleSearch()

# Create a prompt template for the agent
template = "You are a helpful assistant that can chat and search on Google."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# Define a function to handle user input and agent response
def chat_and_search(user_input):
    # Format the prompt with the user input
    prompt = chat_prompt.format_prompt(text=user_input).to_string()
    
    # Call the chat model with the prompt
    response = chat(prompt)
    
    # Check if the response contains an action tag for Google search
    if response.actionTag == "google_search":
        # Extract the query from the action input
        query = response.actionInput
        
        # Call the Google search tool with the query
        results = search(query)
        
        # Format the results as a list of titles and URLs
        results_list = []
        for result in results:
            title = result["title"]
            url = result["url"]
            results_list.append(f"{title}: {url}")
        
        # Join the results with newlines and return as a message
        results_message = "\n".join(results_list)
        return AIMessage(content=results_message)
    
    # Otherwise, return the response as is
    else:
        return response

# Test the function with some user inputs
user_input_1 = "Tell me a joke."
response_1 = chat_and_search(user_input_1)
print(response_1.content)

user_input_2 = "Search for LangChain on Google."
response_2 = chat_and_search(user_input_2)
print(response_2.content)
