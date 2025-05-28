import os
import unittest
from unittest.mock import patch, MagicMock, mock_open
from rag_app.create import VectorStoreCreator


class TestVectorStoreCreator(unittest.TestCase):
    def setUp(self):
        self.test_vector_store_dir = "test_vector_store"
        self.test_source_dir = "test_sources"
        self.test_embeddings_model_path = "test_model.gguf"
        
    def tearDown(self):
        pass
        
    def test_init_with_defaults(self):
        with patch.dict(os.environ, {
            "SOURCE_DIR": "env_sources",
            "EMBEDDINGS_MODEL_PATH": "env_model.gguf",
            "SOURCE_TYPE": "pdf"
        }):
            creator = VectorStoreCreator()
            self.assertEqual(creator.vector_store_dir, "vector_store_db")
            self.assertEqual(creator.source_dir, "env_sources")
            self.assertEqual(creator.embeddings_model_path, "env_model.gguf")
            self.assertEqual(creator.source_type, "pdf")
    
    def test_init_with_params(self):
        creator = VectorStoreCreator(
            vector_store_dir=self.test_vector_store_dir,
            source_dir=self.test_source_dir,
            embeddings_model_path=self.test_embeddings_model_path,
            source_type="csv"
        )
        self.assertEqual(creator.vector_store_dir, self.test_vector_store_dir)
        self.assertEqual(creator.source_dir, self.test_source_dir)
        self.assertEqual(creator.embeddings_model_path, self.test_embeddings_model_path)
        self.assertEqual(creator.source_type, "csv")
    
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_check_vector_store_exists_true(self, mock_listdir, mock_isdir):
        mock_isdir.return_value = True
        mock_listdir.return_value = ['index.faiss', 'index.pkl']
        
        creator = VectorStoreCreator(vector_store_dir=self.test_vector_store_dir)
        self.assertTrue(creator.check_vector_store_exists())
        
        mock_isdir.assert_called_once_with(self.test_vector_store_dir)
        mock_listdir.assert_called_once_with(self.test_vector_store_dir)
    
    @patch('os.path.isdir')
    def test_check_vector_store_exists_false_no_dir(self, mock_isdir):
        mock_isdir.return_value = False
        
        creator = VectorStoreCreator(vector_store_dir=self.test_vector_store_dir)
        self.assertFalse(creator.check_vector_store_exists())
        
        mock_isdir.assert_called_once_with(self.test_vector_store_dir)
    
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_check_vector_store_exists_false_empty_dir(self, mock_listdir, mock_isdir):
        mock_isdir.return_value = True
        mock_listdir.return_value = []
        
        creator = VectorStoreCreator(vector_store_dir=self.test_vector_store_dir)
        self.assertFalse(creator.check_vector_store_exists())
        
        mock_isdir.assert_called_once_with(self.test_vector_store_dir)
        mock_listdir.assert_called_once_with(self.test_vector_store_dir)
    
    @patch('rag_app.create.DirectoryLoader')
    def test_load_pdf_documents(self, mock_loader_class):
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.load.return_value = ["doc1", "doc2"]
        
        creator = VectorStoreCreator(source_dir=self.test_source_dir, source_type="pdf")
        docs = creator.load_pdf_documents()
        
        self.assertEqual(docs, ["doc1", "doc2"])
        mock_loader_class.assert_called_once()
        mock_loader.load.assert_called_once()
    
    @patch('rag_app.create.DirectoryLoader')
    def test_load_csv_documents(self, mock_loader_class):
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.load.return_value = ["doc1", "doc2"]
        
        creator = VectorStoreCreator(source_dir=self.test_source_dir, source_type="csv")
        docs = creator.load_csv_documents()
        
        self.assertEqual(docs, ["doc1", "doc2"])
        mock_loader_class.assert_called_once()
        mock_loader.load.assert_called_once()
    
    def test_load_documents_unsupported_type(self):
        creator = VectorStoreCreator(source_type="unsupported")
        with self.assertRaises(ValueError) as context:
            creator.load_documents()
        
        self.assertIn("Unsupported source type", str(context.exception))
    
    @patch('rag_app.create.RecursiveCharacterTextSplitter')
    def test_split_documents(self, mock_splitter_class):
        mock_splitter = MagicMock()
        mock_splitter_class.return_value = mock_splitter
        mock_splitter.split_documents.return_value = ["chunk1", "chunk2", "chunk3"]
        
        creator = VectorStoreCreator()
        docs = ["doc1", "doc2"]
        chunks = creator.split_documents(docs)
        
        self.assertEqual(chunks, ["chunk1", "chunk2", "chunk3"])
        mock_splitter_class.assert_called_once()
        mock_splitter.split_documents.assert_called_once_with(docs)
    
    @patch('rag_app.create.LlamaCppEmbeddings')
    @patch('rag_app.create.FAISS')
    def test_create_vector_store(self, mock_faiss, mock_embeddings_class):
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        
        mock_vs = MagicMock()
        mock_faiss.from_documents.return_value = mock_vs
        
        with patch.dict(os.environ):
            creator = VectorStoreCreator(
                vector_store_dir=self.test_vector_store_dir,
                embeddings_model_path=self.test_embeddings_model_path
            )
            docs = ["doc1", "doc2"]
            vs = creator.create_vector_store(docs)
            
            self.assertEqual(vs, mock_vs)
            mock_embeddings_class.assert_called_once_with(
                model_path=self.test_embeddings_model_path,
                verbose=False
            )
            mock_faiss.from_documents.assert_called_once_with(docs, mock_embeddings)
            mock_vs.save_local.assert_called_once_with(self.test_vector_store_dir)
    
    @patch.object(VectorStoreCreator, 'check_vector_store_exists')
    @patch.object(VectorStoreCreator, 'load_documents')
    @patch.object(VectorStoreCreator, 'split_documents')
    @patch.object(VectorStoreCreator, 'create_vector_store')
    def test_create_new_vector_store(self, mock_create_vs, mock_split, mock_load, mock_check):
        mock_check.return_value = False
        mock_load.return_value = ["doc1", "doc2"]
        mock_split.return_value = ["chunk1", "chunk2", "chunk3"]
        mock_vs = MagicMock()
        mock_vs.index.ntotal = 3
        mock_create_vs.return_value = mock_vs
        
        creator = VectorStoreCreator(
            vector_store_dir=self.test_vector_store_dir,
            source_dir=self.test_source_dir,
            source_type="pdf"
        )
        result = creator.create()
        
        self.assertEqual(result, mock_vs)
        mock_check.assert_called_once()
        mock_load.assert_called_once()
        mock_split.assert_called_once_with(["doc1", "doc2"])
        mock_create_vs.assert_called_once_with(["chunk1", "chunk2", "chunk3"])
    
    @patch.object(VectorStoreCreator, 'check_vector_store_exists')
    @patch.object(VectorStoreCreator, 'load_documents')
    def test_create_existing_vector_store_no_force(self, mock_load, mock_check):
        mock_check.return_value = True
        
        creator = VectorStoreCreator(vector_store_dir=self.test_vector_store_dir)
        result = creator.create(force=False)
        
        self.assertIsNone(result)
        mock_check.assert_called_once()
        mock_load.assert_not_called()
    
    @patch.object(VectorStoreCreator, 'check_vector_store_exists')
    @patch.object(VectorStoreCreator, 'load_documents')
    @patch.object(VectorStoreCreator, 'split_documents')
    @patch.object(VectorStoreCreator, 'create_vector_store')
    def test_create_existing_vector_store_with_force(self, mock_create_vs, mock_split, mock_load, mock_check):
        mock_check.return_value = True
        mock_load.return_value = ["doc1", "doc2"]
        mock_split.return_value = ["chunk1", "chunk2", "chunk3"]
        mock_vs = MagicMock()
        mock_vs.index.ntotal = 3
        mock_create_vs.return_value = mock_vs
        
        creator = VectorStoreCreator(
            vector_store_dir=self.test_vector_store_dir,
            source_dir=self.test_source_dir
        )
        result = creator.create(force=True)
        
        self.assertEqual(result, mock_vs)
        mock_check.assert_called_once()
        mock_load.assert_called_once()
        mock_split.assert_called_once_with(["doc1", "doc2"])
        mock_create_vs.assert_called_once_with(["chunk1", "chunk2", "chunk3"])


if __name__ == '__main__':
    unittest.main()