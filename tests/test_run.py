import os
import unittest
from unittest.mock import patch, MagicMock, call
from flask import Flask
from rag_app.run import RagApplication


class TestRagApplication(unittest.TestCase):
    def setUp(self):
        self.test_vector_store_dir = "test_vector_store"
        self.app_instance = RagApplication()
        self.flask_app = self.app_instance.app
        self.test_client = self.flask_app.test_client()
        
    def tearDown(self):
        pass
    
    def test_init_with_defaults(self):
        with patch.dict(os.environ, {"SOURCE_TYPE": "csv"}):
            app = RagApplication()
            self.assertEqual(app.source_type, "csv")
            self.assertEqual(app.vector_store_dir, "vector_store_db")
            self.assertIsNotNone(app.app)
            self.assertIsNone(app.qa_chain)
    
    def test_init_with_params(self):
        app = RagApplication(source_type="pdf", vector_store_dir=self.test_vector_store_dir)
        self.assertEqual(app.source_type, "pdf")
        self.assertEqual(app.vector_store_dir, self.test_vector_store_dir)
        self.assertIsNotNone(app.app)
        self.assertIsNone(app.qa_chain)
    
    @patch('rag_app.run.VectorStoreCreator')
    @patch.object(RagApplication, '_setup_qa_chain')
    @patch.object(RagApplication, '_setup_routes')
    def test_initialize(self, mock_setup_routes, mock_setup_qa_chain, mock_creator_class):
        mock_creator = MagicMock()
        mock_creator_class.return_value = mock_creator
        mock_qa_chain = MagicMock()
        mock_setup_qa_chain.return_value = mock_qa_chain
        
        app = RagApplication(source_type="pdf", vector_store_dir=self.test_vector_store_dir)
        app.initialize(force_create=True)
        
        mock_creator_class.assert_called_once_with(
            source_type="pdf", 
            vector_store_dir=self.test_vector_store_dir
        )
        mock_creator.create.assert_called_once_with(force=True)
        mock_setup_qa_chain.assert_called_once()
        mock_setup_routes.assert_called_once()
        self.assertEqual(app.qa_chain, mock_qa_chain)
    
    @patch('rag_app.run.LlamaCppEmbeddings')
    def test_load_embeddings(self, mock_embeddings_class):
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        
        with patch.dict(os.environ, {"EMBEDDINGS_MODEL_PATH": "test_model.gguf"}):
            app = RagApplication()
            result = app._load_embeddings()
            
            self.assertEqual(result, mock_embeddings)
            mock_embeddings_class.assert_called_once_with(
                model_path="test_model.gguf",
                verbose=False
            )
    
    @patch.object(RagApplication, '_load_embeddings')
    @patch('rag_app.run.FAISS')
    def test_load_vector_store(self, mock_faiss, mock_load_embeddings):
        mock_embeddings = MagicMock()
        mock_load_embeddings.return_value = mock_embeddings
        mock_vs = MagicMock()
        mock_faiss.load_local.return_value = mock_vs
        
        app = RagApplication(vector_store_dir=self.test_vector_store_dir)
        result = app._load_vector_store()
        
        self.assertEqual(result, mock_vs)
        mock_load_embeddings.assert_called_once()
        mock_faiss.load_local.assert_called_once_with(
            self.test_vector_store_dir,
            mock_embeddings,
            allow_dangerous_deserialization=True
        )
    
    @patch('rag_app.run.LlamaCpp')
    def test_load_llm(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        
        with patch.dict(os.environ, {
            "LLM_MODEL_PATH": "test_llm.gguf",
            "TEMPERATURE": "0.5",
            "MAX_TOKENS": "1000",
            "TOP_P": "0.8",
            "TOP_K": "30",
            "REPEAT_PENALTY": "1.2",
            "GPU_LAYERS": "2",
            "GPU_BATCH_SIZE": "128",
            "CPU_THREADS": "4",
            "N_CTX": "2048",
            "GRAMMAR_PATH": "test_grammar.gbnf"
        }):
            app = RagApplication()
            result = app._load_llm()
            
            self.assertEqual(result, mock_llm)
            mock_llm_class.assert_called_once_with(
                model_path="test_llm.gguf",
                temperature=0.5,
                max_tokens=1000,
                top_p=0.8,
                top_k=30,
                repeat_penalty=1.2,
                n_gpu_layers=2,
                n_batch=128,
                n_threads=4,
                f16_kv=True,
                use_mlock=True,
                verbose=False,
                n_ctx=2048,
                seed=42,
                use_mmap=True,
                stop=["Human:", "Context information is below:"],
                grammar_path="test_grammar.gbnf"
            )
    
    def test_create_qa_prompt(self):
        app = RagApplication()
        prompt = app._create_qa_prompt()
        
        self.assertIn("context", prompt.input_variables)
        self.assertIn("question", prompt.input_variables)
        self.assertIn("Context information is below:", prompt.template)
        self.assertIn("Given the context information and no prior knowledge", prompt.template)
    
    @patch.object(RagApplication, '_load_vector_store')
    @patch.object(RagApplication, '_load_llm')
    @patch.object(RagApplication, '_create_qa_prompt')
    @patch('rag_app.run.RetrievalQA')
    def test_setup_qa_chain(self, mock_retrieval_qa, mock_create_prompt, mock_load_llm, mock_load_vs):
        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever
        mock_load_vs.return_value = mock_vs
        
        mock_llm = MagicMock()
        mock_load_llm.return_value = mock_llm
        
        mock_prompt = MagicMock()
        mock_create_prompt.return_value = mock_prompt
        
        mock_qa_chain = MagicMock()
        mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain
        
        app = RagApplication()
        result = app._setup_qa_chain()
        
        self.assertEqual(result, mock_qa_chain)
        mock_load_vs.assert_called_once()
        mock_load_llm.assert_called_once()
        mock_create_prompt.assert_called_once()
        mock_vs.as_retriever.assert_called_once()
        mock_retrieval_qa.from_chain_type.assert_called_once_with(
            llm=mock_llm,
            chain_type="stuff",
            retriever=mock_retriever,
            chain_type_kwargs={"prompt": mock_prompt},
            tags=None
        )
    
    def test_handle_query_success(self):
        mock_qa_chain = MagicMock()
        mock_qa_chain.invoke.return_value = {"result": " RAG is a retrieval-augmented generation model. "}

        self.app_instance.qa_chain = mock_qa_chain

        self.app_instance._setup_routes()

        with self.flask_app.test_request_context('/query', method='POST', 
                                               json={"question": "What is RAG?"}):
            response = self.app_instance._handle_query()

            self.assertIn('data', response.json)
            self.assertEqual(response.json['data']['question'], "What is RAG?")
            self.assertEqual(response.json['data']['answer'], "RAG is a retrieval-augmented generation model.")
            mock_qa_chain.invoke.assert_called_once_with({"query": "What is RAG?"}, config={})
    
    def test_handle_query_missing_question(self):
        self.app_instance._setup_routes()
        
        with self.flask_app.test_request_context('/query', method='POST', json={}):
            response = self.app_instance._handle_query()
            
            self.assertEqual(response[1], 400)  # Check status code
            self.assertEqual(response[0].json, {"error": "Missing 'question' field in request body"})
    
    def test_handle_query_exception(self):
        mock_qa_chain = MagicMock()
        mock_qa_chain.invoke.side_effect = Exception("Test error")
        
        self.app_instance.qa_chain = mock_qa_chain
        
        self.app_instance._setup_routes()
        
        with self.flask_app.test_request_context('/query', method='POST', 
                                               json={"question": "What is RAG?"}):
            response = self.app_instance._handle_query()
            
            self.assertEqual(response[1], 500)  # Check status code
            self.assertEqual(response[0].json, {"error": "Test error"})
            mock_qa_chain.invoke.assert_called_once_with({"query": "What is RAG?"}, config={})
    
    def test_setup_routes(self):
        app = RagApplication()
        mock_flask_app = MagicMock()
        app.app = mock_flask_app
        
        app._setup_routes()
        
        mock_flask_app.route.assert_called_once_with('/query', methods=['POST'])
    
    def test_run(self):
        with patch.dict(os.environ, {"FLASK_PORT": "5000"}):
            with patch.object(Flask, 'run') as mock_run:
                app = RagApplication()
                app.run(host='127.0.0.1', debug=True)
                
                mock_run.assert_called_once_with(host='127.0.0.1', port=5000, debug=True)
    
    def test_run_with_default_port(self):
        with patch.dict(os.environ, {}):
            with patch.object(Flask, 'run') as mock_run:
                app = RagApplication()
                app.run()
                
                mock_run.assert_called_once_with(host='0.0.0.0', port=8080, debug=False)


if __name__ == '__main__':
    unittest.main()