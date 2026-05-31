import os
import unittest
import threading
import time
from unittest.mock import patch, MagicMock
from openai import RateLimitError
from key_manager import APIKeyManager, RotationalOpenAIClient


class TestRotationalClient(unittest.TestCase):

    def setUp(self):
        # Set up mock key variables in environment
        os.environ["KEY_1"] = "mock_key_one_xxxxxx"
        os.environ["KEY_2"] = "mock_key_two_yyyyyy"
        os.environ["KEY_3"] = "mock_key_three_zzzzzz"

    @patch('key_manager.OpenAI')
    def test_key_rotation_on_rate_limit(self, mock_openai_class):
        """Verify that when the active key receives a 429 RateLimitError, the manager automatically rotates to the next key."""
        # 1. Setup mock OpenAI client instances
        mock_client_one = MagicMock()
        mock_client_two = MagicMock()

        # Mock a 429 response structure
        mock_http_response = MagicMock()
        mock_http_response.status_code = 429
        
        # Make the first client raise a 429 RateLimitError
        mock_client_one.chat.completions.create.side_effect = RateLimitError(
            message="429 Rate limit exceeded for KEY_1",
            response=mock_http_response,
            body=None
        )

        # Make the second client return a successful dummy object
        dummy_response = MagicMock()
        dummy_response.choices = [MagicMock(message=MagicMock(content="Hello from Key 2"))]
        mock_client_two.chat.completions.create.return_value = dummy_response

        # Instruct patch to return client_one, then client_two
        mock_openai_class.side_effect = [mock_client_one, mock_client_two]

        # Initialize the Rotational client with our mock keys
        client = RotationalOpenAIClient(key_patterns=["KEY_1", "KEY_2"], cooldown_duration=30.0)

        # 2. Trigger completion call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Ping"}]
        )

        # 3. Assertions
        # Ensure we successfully fallback to the second key and fetch its output
        self.assertEqual(response.choices[0].message.content, "Hello from Key 2")
        
        # Verify that KEY_1 was correctly placed on cooldown
        self.assertIn("mock_key_one_xxxxxx", client.key_manager.cooldowns)
        self.assertGreater(client.key_manager.cooldowns["mock_key_one_xxxxxx"], time.time())

    @patch('key_manager.OpenAI')
    def test_thread_safety_concurrency(self, mock_openai_class):
        """Simulate multiple concurrent execution threads querying the manager to prove lock safety."""
        mock_client = MagicMock()
        dummy_response = MagicMock()
        dummy_response.choices = [MagicMock(message=MagicMock(content="Concurrent response ok"))]
        mock_client.chat.completions.create.return_value = dummy_response
        mock_openai_class.return_value = mock_client

        # Create rotational client with three keys
        client = RotationalOpenAIClient(key_patterns=["KEY_1", "KEY_2", "KEY_3"], cooldown_duration=10.0)

        results = []
        def worker_thread():
            for _ in range(5):
                res = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Thread check"}]
                )
                results.append(res)

        # Spin up 4 concurrent threads making 5 requests each
        threads = [threading.Thread(target=worker_thread) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Total successful responses should equal 20 (4 threads * 5 loops)
        self.assertEqual(len(results), 20)


if __name__ == "__main__":
    print("--------------------------------------------------")
    print("Running API Key Manager unit and concurrency tests...")
    print("--------------------------------------------------")
    unittest.main()
