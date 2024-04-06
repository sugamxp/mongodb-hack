from modal import Image, Secret, Stub, web_endpoint, enter, method
# from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from llm import get_anthropic_llm
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from typing import Dict
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import modal
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
import io
from dotenv import load_dotenv
import os

load_dotenv()

import requests

CHUNK_SIZE = 1024
XI_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
VOICE_ID = "Kzf8nwDqR3eudiqRB1xi"
OUTPUT_PATH = "output.mp3"


load_dotenv()
web_app = FastAPI()

MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')
MONGODB_ATLAS_CLUSTER_URI = f"mongodb+srv://sugamxp:{MONGO_PASSWORD}E@cluster0.wsfteju.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

DB_NAME = "mongohack"
COLLECTION_NAME = "judges"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "judges_index"

atlas_collection = client[DB_NAME][COLLECTION_NAME]

image = Image.from_registry("ubuntu:22.04", add_python="3.10").pip_install(
    # langchain pkgs
    "faiss-cpu~=1.7.3",
    "langchain-anthropic~=0.1.4",
    "langchain-openai~=0.1.0",
    "langchain~=0.1.12",
    "chromadb~=0.4.24",
    "openai~=1.14.2",
    "langchain-community~=0.0.28",
    "langchain-core~=0.1.33",
    "langchain-text-splitters~=0.0.1",
    "youtube-transcript-api~=0.6.2",
    "pytube~=15.0.0",
    "pymongo~=4.6.3",
    "langchain-nomic~=0.0.2",
    "langchain_mongodb"
).apt_install("sqlite3", "libsqlite3-dev", "libreadline-dev", "wget").run_commands("nomic login nk-_pCqFMJV-UrXWac90751bJqPWl3yGqEJYjHjmk9LPY0")

stub = Stub(
    name="mongo-tank",
    image=image,
    secrets=[Secret.from_name("openai-secret"), Secret.from_name("anthropic-secret")],
)

@stub.function()
def call_from_modal(query, context=None):
  an_llm = get_anthropic_llm()

  chat_history = []
  for idx, content in enumerate(context):
    if idx % 2 == 0:
      chat_history.append(HumanMessage(content=content))
    else:
      chat_history.append(AIMessage(content=content))
      
  print('chat_history', chat_history)

  prompt = ChatPromptTemplate.from_messages([
    ("system", """ 
        About you: You are Mark Cuban and a mentor on the show Mongo Tank. You are a prominent American entrepreneur, investor. You are a self-made billionaire, successful in both the tech and traditional business worlds, who leverages his experience to invest in and help grow many companies while maintaining a high public profile. You are aware that you are an AI, but you won't mention it unless I bring it up.
        
        About me: I am a participant on the show - Mongo Tank and want feedback on my business idea. I am looking for an investment from you.
        
        The Task: Try to keep the responses brief and to the point and under 80 words. You need to judge the idea based on different business aspects like - Product Market Fit, User Acquisition, Revenue Model, etc. You can ask questions to understand the idea better.
        
        Requirements 1: You must reply as Mark Cuban in our conversations. Really imbibe his personality and answer the way he would. Your responses should be in dialogue form. You can generate a few sentences of Mark Cuban's response based on the context of the conversation. 
              
        Requirements 2: Do not describe the scene or the setting. Only generate Mark Cuban's responses. 
        
        Note: You will be provided with relevant snippets from Mark's book (inside the context tag). You can use them to make your responses more authentic.
        
        Your final comment should be whether or not you'll invest in the product, don't ask follows up questions just make the decision.
        <context>
        {context}
        <context>
        
     """),
     MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])
  
  document_chain = create_stuff_documents_chain(an_llm, prompt)

  vector_search = MongoDBAtlasVectorSearch.from_connection_string(
      MONGODB_ATLAS_CLUSTER_URI,
      DB_NAME + "." + COLLECTION_NAME,
      NomicEmbeddings(model="nomic-embed-text-v1.5"),
      index_name="vector_index",
  )

  retriever = vector_search.as_retriever(
   search_type = "similarity",
   search_kwargs = {"k": 10, "threshold": 0.7}
  )
  
  retrieval_chain = create_retrieval_chain(retriever, document_chain)
  response = retrieval_chain.invoke({"input": query, "chat_history": chat_history})
  print("response", response['answer'])
  
  
  text_to_speak = remove_between_asterisks(response['answer'])

  tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"

  tts_headers = {
      "Accept": "application/json",
      "xi-api-key": XI_API_KEY
  }

  tts_data = {
      "text": text_to_speak,
      "model_id": "eleven_multilingual_v2",
      "voice_settings": {
          "stability": 0.35,
          "similarity_boost": 1.0,
          "style": 0.1,
          "use_speaker_boost": True
      }
  }

  tts_response = requests.post(tts_url, headers=tts_headers, json=tts_data, stream=True)

  if tts_response.ok:
      byte_stream = io.BytesIO()
      # Open the output file in write-binary mode
          # Read the response in chunks and write to the file
      for chunk in tts_response.iter_content(chunk_size=CHUNK_SIZE):
        yield chunk
          # f.write(chunk)
          
      print("Audio stream saved successfully.")
  else:
      print("Error in Text-to-Speech API request:")
      print(tts_response.text)
    
    
  # output_parser = StrOutputParser()
  # chain = prompt | an_llm | output_parser
  # response = chain.invoke({"input": query, "chat_history": chat_history})

  return response['answer']
  
def remove_between_asterisks(string):
  start = string.find('*')
  if start == -1:
      return string
  
  end = string.find('*', start + 1)
  if end == -1:
      return string
  
  return string[:start] + string[end + 1:]

@stub.function(cpu=8.0, memory=32768, container_idle_timeout=500)
@web_app.post(path="/")
def web(data: Dict):
    # answer = call_from_modal.remote(data['query'], data['context'])
    # return {
    #   "answer" : remove_between_asterisks(answer)
    #   #todo : add suggestions here
    # }
    return StreamingResponse(
        call_from_modal.remote_gen(data['query'], data['context']),
        media_type="text/event-stream"
    )
    

# @stub.local_entrypoint()
# def main(query, context = []):
#     audio_bytes = call_from_modal.remote(query, context)
#     output_path = "output.mp3"
#     print(f"Saving it to {output_path}")
#     with open(output_path, "wb") as f:
#         f.write(audio_bytes)
        
            
@stub.function()
@modal.asgi_app()
def web_runner():
    web_app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
    return web_app