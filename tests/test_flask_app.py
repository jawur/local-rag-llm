import unittest
from unittest.mock import patch, MagicMock
from rag_app.run import create_app


class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        self.mock_rag_app = MagicMock()
        self.mock_rag_app.query.return_value = {
            "question": "What is RAG?",
            "answer": "RAG is a retrieval-augmented generation model."
        }
        self.app = create_app(self.mock_rag_app)
        self.client = self.app.test_client()
        
    def test_handle_query_success(self):
        response = self.client.post('/api/query', json={"question": "What is RAG?"})
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["data"]["question"], "What is RAG?")
        self.assertEqual(response.json["data"]["answer"], "RAG is a retrieval-augmented generation model.")
        self.mock_rag_app.query.assert_called_once_with("What is RAG?")
    
    def test_handle_query_missing_question(self):
        response = self.client.post('/api/query', json={})
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"], "Missing 'question' field in request body")
        self.mock_rag_app.query.assert_not_called()
    
    def test_handle_query_exception(self):
        self.mock_rag_app.query.side_effect = Exception("Test error")
        
        response = self.client.post('/api/query', json={"question": "What is RAG?"})
        
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json["error"], "Test error")
        self.mock_rag_app.query.assert_called_once_with("What is RAG?")
    
    def test_serve_openapi_spec(self):
        with patch('rag_app.run.send_file') as mock_send_file:
            mock_send_file.return_value = "openapi spec"
            self.client.get('/api/openapi.yaml')
            mock_send_file.assert_called_once_with('../openapi.yaml', mimetype='text/yaml')
    
    def test_swagger_ui_endpoint(self):
        response = self.client.get('/api/docs/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'swagger-ui', response.data)


if __name__ == '__main__':
    unittest.main()