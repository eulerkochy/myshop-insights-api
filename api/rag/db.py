import dotenv
dotenv.load_dotenv()

from pymongo import MongoClient
import os

def get_rag_database(db_name): 
  # Provide the mongodb atlas url to connect python to mongodb using pymongo
  CONNECTION_STRING = os.environ.get('MONGODB_URI')
  client = MongoClient(CONNECTION_STRING)
  return client[db_name]