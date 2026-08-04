from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    max_new_tokens=200,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    input_variables=['topic'],
    template="Suggest a catchy blog title about {topic}."
)

topic = input('Enter a topic')

formatted_prompt = prompt.format(topic=topic)

response = model.invoke(formatted_prompt)

print("Generated Blog Title:", response.content)