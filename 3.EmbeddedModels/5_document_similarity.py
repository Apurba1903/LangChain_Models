from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(
    model='text-embedding-3-large', 
    dimensions=300
)


documents = [
    "Documents Strings"
]

query = "Question about Document"


doc_embeddings = embedding.embed_documents(documents)

query_embeddings = embedding.embed_query(query)


scores = cosine_similarity([query_embeddings], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]


print(query)
print(documents[index])
print("Similarity Score is : ", score)