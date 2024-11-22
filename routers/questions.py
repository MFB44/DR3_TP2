from fastapi import APIRouter
from pydantic import BaseModel
from transformers import pipeline
from langchain_community.llms import FakeListLLM
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
load_dotenv('./env')

router = APIRouter()

class AutoCompleteModel(BaseModel):
    phrase: str

class TranslationModel(BaseModel):
    phrase: str

class FakeLLMModel(BaseModel):
    prompt: str

# [Parte 1: Questão 1] Função para gerar uma resposta com base em uma mensagem
def generate_response(message: str):
    generator = pipeline("text-generation", model="gpt2-large")
    return generator(message)

# [Parte 1: Questão 2] Função para traduzir uma mensagem de inglês para francês
def translate_eng_franc(message: str):
    translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
    return translator(message)

# [Parte 2: Questão 1] Função para gerar uma resposta com base em FakeLLM
def use_fake_llm(prompt):
    """
    [Parte 2: Questão 1]
    Using FakeLLM to generate responses
    """
    fake_llm = FakeListLLM(responses=[
        "The sky is blue.",
        "The grass is green.",
        "Water is wet.",
        "Fire is hot.",
        "The Earth orbits the Sun.",
        "Humans need oxygen to breathe."
    ])

    return(fake_llm.invoke(prompt))

# [Parte 2: Questão 2] Função para traduzir uma mensagem de inglês para francês com API OpenAI
def use_openai_api(prompt):
    """
    [Parte 2: Questão 2]
    Using OpenAI API to translate text
    """
    template = ChatPromptTemplate([
        ("system", "You are an english to fench translator. Return the translations to a JSON format."),
        ("user", "Translate this: {text}")
    ])
    llm = ChatOpenAI(model="gpt-40", api_key=os.environ["OPENAI_KEY"])
    response = llm.invoke(template.format_messages(text=prompt))
    return(response.content)

# [Parte 2: Questão 3] Função para traduzir uma mensagem de inglês para alemão
def translate_eng_german(message: str):
    translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
    return translator(message)

# [Parte 1: Questão 1] Rota para gerar uma resposta com base em uma mensagem
@router.post("/autocomplete/")
async def autocomplete(body:AutoCompleteModel):
    response = generate_response(body.phrase)
    return {"assistant": response}

# [Parte 1: Questão 2] Rota para traduzir uma mensagem de inglês para francês
@router.post("/translate_french/")
async def translate(body:TranslationModel):
    response = translate_eng_franc(body.phrase)
    return {"assistant": response}

# [Parte 2: Questão 1] Rota para gerar uma resposta com base em FakeLLM
@router.post("/fake_llm/")
async def fake_llm(body:FakeLLMModel):
    response = use_fake_llm(body.prompt)
    return {"assistant": response}

# [Parte 2: Questão 2] Rota para traduzir uma mensagem de inglês para francês com API OpenAI
@router.post("/openai_api/")
async def openai_api(body:TranslationModel):
    response = use_openai_api(body.phrase)
    return {"assistant": response}

# [Parte 2: Questão 3] Rota para traduzir uma mensagem de inglês para alemão
@router.post("/translate_german/")
async def translate_german(body:TranslationModel):
    response = translate_eng_german(body.phrase)
    return {"assistant": response}




