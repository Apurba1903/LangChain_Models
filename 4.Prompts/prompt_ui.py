from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()
model = ChatOpenAI()

st.header('Research Tool')

paper_input = st.selectbox(
    "Select Research Paper Name", 
    [
        "Select...",
        "Attention Is All You Need",
        "BERT: Pre-Training of Deep Bi-Directional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 Paragraph)",
        "Medium (3-5 Paragraph)",
        "Long (Detailed Explanation)"
    ]
)

template = load_prompt('C:/Users/ACER/Desktop/DSMP1/LangChain/4.Prompts/template.json')


if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)




