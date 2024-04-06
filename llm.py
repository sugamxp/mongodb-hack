from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv
load_dotenv()

def get_anthropic_llm():
  return ChatAnthropic(model='claude-3-opus-20240229', 
                       temperature=0.5,
                       anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
                       max_tokens_to_sample=800)

def get_faster_model():
  return ChatAnthropic(model='claude-3-haiku-20240307', 
                       temperature=0,
                       anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'))

def get_openai_llm():
  return ChatOpenAI(model='gpt-4-0125-preview', temperature=0.5, api_key=os.getenv('OPENAI_API_KEY'))
