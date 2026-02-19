from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.7,
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)

# Schema
class Review(TypedDict):
    
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(""" The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this. """)

print(result)