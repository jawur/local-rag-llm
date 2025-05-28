import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader, 
    UnstructuredPDFLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredHTMLLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

class VectorStoreCreator:
    def __init__(self, vector_store_dir="vector_store_db", source_dir=None, embeddings_model_path=None, source_type=None):
        """Initialize the VectorStoreCreator.
        
        Args:
            vector_store_dir (str): Directory where the vector store will be saved
            source_dir (str): Directory containing the files to process
            embeddings_model_path (str): Path to the embeddings model
            source_type (str): Type of source files to process (pdf, csv, json, html, txt)
        """
        self.vector_store_dir = vector_store_dir
        self.source_dir = source_dir or os.getenv("SOURCE_DIR")
        self.embeddings_model_path = embeddings_model_path or os.getenv("EMBEDDINGS_MODEL_PATH")
        self.source_type = source_type or os.getenv("SOURCE_TYPE", "pdf").lower()
        
    def check_vector_store_exists(self):
        return os.path.isdir(self.vector_store_dir) and os.listdir(self.vector_store_dir)

    def load_documents(self):
        if self.source_type == "pdf":
            return self.load_pdf_documents()
        elif self.source_type == "csv":
            return self.load_csv_documents()
        elif self.source_type == "json":
            return self.load_json_documents()
        elif self.source_type == "html":
            return self.load_html_documents()
        elif self.source_type == "txt":
            return self.load_txt_documents()
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}. Supported types: pdf, csv, json, html, txt")
    
    def load_pdf_documents(self):
        loader = DirectoryLoader(
            self.source_dir,
            glob="**/*.pdf",
            loader_cls=UnstructuredPDFLoader,
            show_progress=True
        )
        return loader.load()
        
    def load_csv_documents(self):
        loader = DirectoryLoader(
            self.source_dir,
            glob="**/*.csv",
            loader_cls=CSVLoader,
            show_progress=True
        )
        return loader.load()
        
    def load_json_documents(self):
        # For JSONLoader, we need to specify the jq schema to extract text
        # Using a simple approach to extract all text values
        loader = DirectoryLoader(
            self.source_dir,
            glob="**/*.json",
            loader_cls=lambda file_path: JSONLoader(
                file_path=file_path,
                jq_schema=".",
                text_content=False
            ),
            show_progress=True
        )
        return loader.load()
        
    def load_html_documents(self):
        loader = DirectoryLoader(
            self.source_dir,
            glob="**/*.html",
            loader_cls=UnstructuredHTMLLoader,
            show_progress=True
        )
        return loader.load()
        
    def load_txt_documents(self):
        loader = DirectoryLoader(
            self.source_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        return loader.load()

    def split_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(docs)

    def create_vector_store(self, docs):
        embeddings = LlamaCppEmbeddings(
            model_path=self.embeddings_model_path,
            verbose=False
        )

        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(self.vector_store_dir)
        return vector_store

    def create(self, force=False):
        """Create the vector store if it doesn't exist or if force is True.
        
        Args:
            force (bool): If True, create vector store even if it already exists
            
        Returns:
            FAISS: The created vector store instance
        """
        if self.check_vector_store_exists() and not force:
            print(f"Directory '{self.vector_store_dir}' exists and is not empty. Skipping creation.")
            return None

        print(f"Loading {self.source_type.upper()} documents from {self.source_dir}...")
        documents = self.load_documents()
        print(f"Loaded {len(documents)} documents")

        print("Splitting documents...")
        split_docs = self.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks")

        print("Creating vector store...")
        vs = self.create_vector_store(split_docs)
        print(f"Vector store created and saved to {self.vector_store_dir}")
        print(f"Total vectors stored: {vs.index.ntotal}")
        return vs

if __name__ == "__main__":
    creator = VectorStoreCreator()
    creator.create()