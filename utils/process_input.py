import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter


def process_pdf(pdf_docs):

    chunks = list()

    for pdf in pdf_docs:
        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
        )

        docs = text_splitter.create_documents([text])

        try:
            source = pdf.name
        except:
            source = os.path.basename(pdf)

        for doc in docs:
            doc.metadata = {'source': source}
            chunks.append(doc)

    return chunks

