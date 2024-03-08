import json

from langchain.docstore.document import Document

class CustomDocumentLoader:
  def __init__(self, file_path):
    self.file_path = file_path

  def load(self):
    doc_list = []
    with open(self.file_path, 'r') as f:
      data = json.load(f)
      for item in data['data']:
        page_content = item['html_body']
        page_source = item['loc']
        page_title = item['images'][0]['title'] if len(item['images']) > 0 else ""
        doc = Document(page_content=page_content, metadata={"source": page_source, "title": page_title})
        doc_list.append(doc)

    return doc_list