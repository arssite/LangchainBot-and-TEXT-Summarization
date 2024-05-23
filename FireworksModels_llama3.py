from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_community.llms import Ollama
import streamlit as st
import os
#from dotenv import load_dotenv
from langchain_fireworks import Fireworks 
#load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["FIREWORKS_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"]=""
## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)
## streamlit framework
st.title('Langchain Demo With LLAMA2 API')
input_text=st.text_input("Search the topic u want")
# ollama LLAma2 LLm 
#llm=Ollama(model="llama2")
llm = Fireworks(
    #api_key="3pJjPGWVwLfUY4UFS3dMMRtGWRifofZWERtyduEtLP2v36fw",
    model="accounts/fireworks/models/llama-v2-70b-chat",
    max_tokens=256)
llm("Name 3 sports.")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text: 
        st.write(chain.invoke({"question":input_text[6:]}))
