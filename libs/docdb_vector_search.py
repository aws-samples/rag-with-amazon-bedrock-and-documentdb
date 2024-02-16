#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# vim: tabstop=2 shiftwidth=2 softtabstop=2 expandtab

from typing import (
  Any,
  Dict,
  List,
  Optional,
)

from langchain_core.documents import Document
from langchain_community.vectorstores import MongoDBAtlasVectorSearch

class DocumentDBAVectorSearch(MongoDBAtlasVectorSearch):
  def similarity_search(
    self,
    query: str,
    k: int = 4,
    pre_filter: Optional[Dict] = None,
    post_filter_pipeline: Optional[List[Dict]] = None,
    **kwargs: Any,
  ) -> List[Document]:
    """Return Amazon DocumentDB documents most similar to the given query.

    Uses the vectorSearch operator available in Amazon DocumentDB Search.
    For more: https://docs.aws.amazon.com/documentdb/latest/developerguide/vector-search.html

    Args:
        query: Text to look up documents similar to.
        k: (Optional) number of documents to return. Defaults to 4.
        pre_filter: (Optional) dictionary of argument(s) to prefilter document
            fields on.
        post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages
            following the vectorSearch stage.

    Returns:
        List of documents most similar to the query.
    """

    similarity = kwargs.get("similarity", "cosine") # similarity = [ euclidean | cosine | dotProduct ]
    embedding = self._embedding.embed_query(query)
    cursor = self._collection.aggregate([{
      '$search': {
        "vectorSearch": {
          "vector": embedding,
          "path": self._embedding_key,
          "similarity": similarity,
          "k": k
        }
      }
    }])

    docs = []
    for res in cursor:
      text = res.pop(self._text_key)
      del res[self._embedding_key]
      docs.append(Document(page_content=text, metadata=res))
    return docs


  def max_marginal_relevance_search(
    self,
    query: str,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    pre_filter: Optional[Dict] = None,
    post_filter_pipeline: Optional[List[Dict]] = None,
    **kwargs: Any,
  ) -> List[Document]:
    raise NotImplementedError
