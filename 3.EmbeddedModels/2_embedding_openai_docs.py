from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(
    model='text-embedding-3-large', 
    dimensions=32
)

documents = [
    "Dhaka is the capital of Bangladesh",
    "Paris is the capital of France",
    "Munich is the capital of Germany"
]

result = embedding.embed_documents(documents)

print(str(result))