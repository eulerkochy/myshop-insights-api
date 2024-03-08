from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from api.rag.dataloader import CustomDocumentLoader
from api.rag.db import get_rag_database

import os

def get_collection(db_name, collection_name):
  db = get_rag_database(db_name)
  return db[collection_name]

def create_vectorstore(store_name, file):
  collection = get_collection(os.environ.get('MONGODB_DBNAME'), store_name)
  loader = CustomDocumentLoader(file)

  docs = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  splits = text_splitter.split_documents(docs)
  vectorstore = MongoDBAtlasVectorSearch.from_documents(documents=splits, embedding=OpenAIEmbeddings(), collection=collection)
  return vectorstore

# TODO: Create vectorindex programmatically

def get_vectorstore(store_name, index_name):
  collection = get_collection(os.environ.get('MONGODB_DBNAME'), store_name)
  return MongoDBAtlasVectorSearch(collection=collection, embedding=OpenAIEmbeddings(), index_name=index_name)

def query_embeddings(store_name, query):
  vectorstore = get_vectorstore(store_name, os.environ.get('MONGODB_INDEX_NAME'))
  retriever = vectorstore.as_retriever()
  prompt = hub.pull("rlm/rag-prompt")
  llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

  # This is very important because otherwise the embeddings vector will be added in the context
  # leading to token exceed error
  def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)
  
  rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
  )

  rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
  ).assign(answer=rag_chain_from_docs)

  result = rag_chain_with_source.invoke(query)

  answer = result['answer']
  context = result['context']
  sources = []
  for doc in context:
    sources.append(doc.metadata['source'])
  sources = list(set(sources))
  
  return answer, sources