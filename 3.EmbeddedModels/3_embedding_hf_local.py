from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = "Dhaka is the Capital of Bangladesh"

vector = embedding.embed_query(text)

print(str(vector))