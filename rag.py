from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
import json


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="mistral")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. [/INST] </s>  
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )
        self.log_file = "query_logs.json"

    def ingest(self, pdf_file_path: str, original_filename: str = None):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.2,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        
        # Log the PDF file path to the JSON log file
        self._log_pdf_ingestion(pdf_file_path, original_filename)

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        answer = self.chain.invoke(query)

        self._log_query(query, answer)

        return answer
    
    def _log_query(self, query, answer):
        """Saves the query, answer, and similarity scores to a JSON log file."""
        log_entry = {
            "query": query,
            "answer": answer,
        }

        # Read existing logs if the file exists
        try:
            with open(self.log_file, 'r') as file:
                logs = json.load(file)
        except FileNotFoundError:
            logs = []

        # Append the new log entry
        logs.append(log_entry)

        # Write the updated logs back to the file
        with open(self.log_file, 'w') as file:
            json.dump(logs, file, indent=4)

    def _log_pdf_ingestion(self, pdf_file_path, original_filename):
        """Logs the ingestion of a new PDF file to the JSON log file."""
        log_entry = {
            "ingested_pdf": pdf_file_path,
            "original_filename": original_filename,
            "retriever_k": self.retriever.search_kwargs["k"],
            "retriever_score_threshold": self.retriever.search_kwargs["score_threshold"]
        }

        # Read existing logs if the file exists
        try:
            with open(self.log_file, 'r') as file:
                logs = json.load(file)
        except FileNotFoundError:
            logs = []

        # Append the new PDF log entry
        logs.append(log_entry)

        # Write the updated logs back to the file
        with open(self.log_file, 'w') as file:
            json.dump(logs, file, indent=4)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
