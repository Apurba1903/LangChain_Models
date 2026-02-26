from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    max_new_tokens=200
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    
    name: str = Field(description='Name of the Person')
    age: int = Field(gt=18, description='Age of the Person')
    city: str = Field(description='Name of the City from the Person belongs to')


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a Fictional {Universe} Villain \n {format_instruction}',
    input_variables=['Universe'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
    
)


chain = template | model | parser

result = chain.invoke({
    'Universe':'Marvel'
})


print(result)