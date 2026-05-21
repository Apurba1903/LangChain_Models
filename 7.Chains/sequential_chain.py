from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed Report on Topic {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a five pointer summary from the following text \n {text}',
    input_variables=['text']
)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    max_new_tokens=200,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Neuclear PowerPlant'})

print(result)

chain.get_graph().print_ascii()