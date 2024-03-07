from typing import Any
from fastapi import APIRouter

router = APIRouter(
  prefix="/rag/mongo",
  tags=["rag_mongo"],
)

from pydantic import BaseModel

class Query(BaseModel):
  store_name: str
  query: str

# TODO: Add a route to create a vectorstore

@router.post("/query")
async def query_store(query: Query) -> Any:
  from rag.mongo import query_embeddings
  answer, sources = query_embeddings(query.store_name, query.query)
  return {"answer": answer, "sources": sources}