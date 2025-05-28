import os
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from rag_app.create import VectorStoreCreator

def import_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")

class RagApplication:
    def __init__(self, source_type=None, vector_store_dir="vector_store_db"):
        """Initialize the RAG application.
        
        Args:
            source_type (str): Type of source files to process
            vector_store_dir (str): Directory where the vector store is saved
        """
        self.source_type = source_type or os.getenv("SOURCE_TYPE", "pdf")
        self.vector_store_dir = vector_store_dir
        self.qa_chain = None
        
    def initialize(self, force_create=False):
        """Initialize the application by creating vector store and setting up QA chain."""
        creator = VectorStoreCreator(source_type=self.source_type, vector_store_dir=self.vector_store_dir)
        creator.create(force=force_create)
        self.qa_chain = self._setup_qa_chain()
        
    def _load_embeddings(self):
        """Load the embeddings model."""
        return LlamaCppEmbeddings(
            model_path=os.getenv("EMBEDDINGS_MODEL_PATH"),
            verbose=False
        )

    def _load_vector_store(self):
        """Load the vector store."""
        embeddings = self._load_embeddings()
        return FAISS.load_local(
            self.vector_store_dir, 
            embeddings,
            allow_dangerous_deserialization=True
        )

    def _load_llm(self):
        """Load the language model."""
        return LlamaCpp(
            model_path=os.getenv("LLM_MODEL_PATH"),
            temperature=float(os.getenv("TEMPERATURE", 0.0)),
            max_tokens=int(os.getenv("MAX_TOKENS", 2000)),
            top_p=float(os.getenv("TOP_P", 0.9)),
            top_k=int(os.getenv("TOP_K", 40)),
            repeat_penalty=float(os.getenv("REPEAT_PENALTY", 1.3)),
            n_gpu_layers=int(os.getenv("GPU_LAYERS", -1)),
            n_batch=int(os.getenv("GPU_BATCH_SIZE", 256)),
            n_threads=int(os.getenv("CPU_THREADS", 6)),
            f16_kv=True,
            use_mlock=True,
            verbose=False,
            n_ctx=int(os.getenv("N_CTX", 4096)),
            seed=42,
            use_mmap=True,
            stop=["Human:", "Context information is below:"],
            grammar_path=os.getenv("GRAMMAR_PATH", None)
        )

    def _create_qa_prompt(self):
        """Create the QA prompt template."""
        template = """You are a helpful assistant that provides accurate information based on the context provided.
If you do not know the answer or the information is not present in the context, do not attempt to make up an answer. 
Instead, respond with "I don't know."

Context information is below:
-----------------------
{context}
-----------------------

Given the context information and no prior knowledge, answer the following question:
{question}

Answer:"""
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def _setup_qa_chain(self):
        """Set up the QA chain."""
        vector_store = self._load_vector_store()
        llm = self._load_llm()
        qa_prompt = self._create_qa_prompt()

        tags = None
        if os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGCHAIN_TRACING_V2") == "true":
            tags = ["rag-llm", "production"]
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": qa_prompt},
            tags=tags
        )

    def query(self, question):
        """Process a query and return the answer.
        
        Args:
            question (str): The question to answer
            
        Returns:
            dict: A dictionary containing the question and answer
        """
        if not question:
            raise ValueError("Question cannot be empty")
            
        metadata = None
        if os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGCHAIN_TRACING_V2") == "true":
            metadata = {
                "endpoint": "/api/query",
                "question_length": len(question),
                "timestamp": str(import_time()) if 'import_time' in globals() else None
            }
        
        result = self.qa_chain.invoke(
            {"query": question},
            config={"metadata": metadata} if metadata else {}
        )
        
        return {
            "question": question,
            "answer": result['result'].strip()
        }