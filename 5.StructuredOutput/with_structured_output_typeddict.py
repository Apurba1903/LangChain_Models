from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatOpenAI()

# Schema
class Review(TypedDict):
    
    summary: str
    sentiment: str


result = model.invoke(""" Review """)