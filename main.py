from api.v1 import rag_mongo
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Union

# Initialize app
app = FastAPI()

# Middleware
origins = ["*"]
app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Routes
app.include_router(rag_mongo.router)