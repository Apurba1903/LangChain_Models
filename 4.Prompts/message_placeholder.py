from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFaceEndpoint

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.7,
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)


# Chat Template
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []

# Load Chat History
with open('C:\\Users\\ACER\\Desktop\\DSMP1\\LangChain\\4.Prompts\\chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# Create Prompt
prompt = chat_template.invoke({
                'chat_history' : chat_history,
                'query': 'Where is my refund?'
            })

print(prompt)