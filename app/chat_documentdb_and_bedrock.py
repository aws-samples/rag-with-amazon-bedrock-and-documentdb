#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# vim: tabstop=2 shiftwidth=2 softtabstop=2 expandtab

import sys
import json
import os

import boto3

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Bedrock
from langchain_community.embeddings import BedrockEmbeddings

from pymongo import MongoClient

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'


MAX_HISTORY_LENGTH = 5

EMBEDDINGS = None
DOC_DB_COLLECTION = None


def _get_credentials(secret_id: str, region_name: str) -> str:
  client = boto3.client('secretsmanager', region_name=region_name)
  response = client.get_secret_value(SecretId=secret_id)
  secrets_value = json.loads(response['SecretString'])
  return secrets_value


def _get_llm(model_id='anthropic.claude-instant-v1', region_name='us-east-1'):
  # configure the properties for Claude
  model_kwargs = {
    "max_tokens_to_sample": 8000,
    "temperature": 0.2,
    "top_k": 250,
    "top_p": 0.9,
    "stop_sequences": ["\\n\\nHuman:"]
  }

  llm = Bedrock(
    region_name=region_name,
    model_id=model_id,
    model_kwargs=model_kwargs
  )

  return llm


def build_chain():
  region = os.environ.get('AWS_REGION', 'us-east-1')
  model_id = os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-instant-v1')

  llm = _get_llm(model_id=model_id, region_name=region)

  # Create a PromptTemplate for the user's question
  question_prompt_template = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="Given this text extracts:\n-----\n{context}\n-----\n and also consider the history of this chat {chat_history}\nPlease answer the following question: {question}",
  )

  # Create an LLMChain
  llm_chain = LLMChain(prompt=question_prompt_template, llm=llm)

  return llm_chain


def _get_or_create_embeddings():
  global EMBEDDINGS

  region = os.environ.get('AWS_REGION', 'us-east-1')

  if EMBEDDINGS is None:
      EMBEDDINGS = BedrockEmbeddings(
        region_name=region
      )
  return EMBEDDINGS


def _get_or_create_collection():
  global DOC_DB_COLLECTION

  if DOC_DB_COLLECTION is None:
    region = os.environ.get('AWS_REGION', 'us-east-1')

    documentdb_secret_name = os.environ['DOCDB_SECRET_NAME']
    creds = _get_credentials(documentdb_secret_name, region)
    USER, PASSWORD = creds['username'], creds['password']

    DOCDB_HOST = os.environ['DOCDB_HOST']
    DB_NAME = os.environ['DB_NAME']
    COLLECTION_NAME = os.environ['COLLECTION_NAME']

    documentdb_client = MongoClient(
      host=DOCDB_HOST,
      port=27017,
      username=USER,
      password=PASSWORD,
      retryWrites=False,
      tls='true',
      tlsCAFile="global-bundle.pem"
    )

    docdb = documentdb_client[DB_NAME]
    DOC_DB_COLLECTION = docdb[COLLECTION_NAME]
  return DOC_DB_COLLECTION


def _get_context_documents(query: str):
  embeddings = _get_or_create_embeddings()
  collection = _get_or_create_collection()

  embedded_query = embeddings.embed_query(query)
  docs = collection.aggregate([{
    '$search': {
      "vectorSearch" : {
        "vector" : embedded_query,
        "path": "embedding",
        "similarity": "euclidean",
        "k": 2
      }
    }
  }])

  return docs


def run_chain(chain, prompt: str, history=[]):
  # Get the user's question and context documents
  question = prompt

  docs = _get_context_documents(prompt)
  context = '\n'.join([doc['text'] for doc in docs])

  # Prepare the input for the LLMChain
  input_data = {
    "context": context,
    "question": question,
    "chat_history": history,
  }

  # Run the LLMChain
  output = chain.invoke(input_data)
  return {'answer': output['text'], 'source_documents': []}


if __name__ == "__main__":
  chat_history = []
  qa = build_chain()
  print(bcolors.OKBLUE + "Hello! How can I help you?" + bcolors.ENDC)
  print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
  print(">", end=" ", flush=True)
  for query in sys.stdin:
    if (query.strip().lower().startswith("new search:")):
      query = query.strip().lower().replace("new search:","")
      chat_history = []
    elif (len(chat_history) == MAX_HISTORY_LENGTH):
      chat_history.pop(0)
    result = run_chain(qa, query, chat_history)
    chat_history.append((query, result["answer"]))
    print(bcolors.OKGREEN + result['answer'] + bcolors.ENDC)
    if 'source_documents' in result:
      print(bcolors.OKGREEN + 'Sources:')
      for d in result['source_documents']:
        print(d.metadata['source'])
    print(bcolors.ENDC)
    print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
    print(">", end=" ", flush=True)
  print(bcolors.OKBLUE + "Bye" + bcolors.ENDC)
