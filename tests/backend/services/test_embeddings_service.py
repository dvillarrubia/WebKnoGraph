import unittest
from unittest import mock
import pandas as pd
import numpy as np

# IMPORTANT: Ensure this import path is correct based on where your
# EmbeddingPipeline class is actually defined.
# If EmbeddingPipeline is defined in src/backend/services/embeddings_service.py, this is correct.
from src.backend.services.embeddings_service import EmbeddingPipeline

# Import other interfaces/configs that the pipeline depends on
from src.backend.config.embeddings_config import EmbeddingConfig
from src.shared.interfaces import ILogger
# You might also need to import the actual classes if you use spec= or autospec= for stricter mocks
# from src.backend.data.embedding_state_manager import EmbeddingStateManager
# from src.backend.data.embeddings_loader import DataLoader
# from src.backend.data.embeddings_saver import DataSaver
# from src.backend.utils.text_processing import TextExtractor
# from src.backend.utils.embedding_generation import EmbeddingGenerator


class TestEmbeddingPipeline(unittest.TestCase):
    def setUp(self):
        # Mock dependencies. Using mock.Mock() for simplicity, but for real-world
        # projects, consider using spec=True or autospec=True for stricter mocking
        # that catches mismatches between the mock and the real object's interface.

        self.mock_config = mock.Mock(spec=EmbeddingConfig)
        self.mock_config.batch_size = 2  # Example batch size for testing

        self.mock_logger = mock.Mock(spec=ILogger)
        self.mock_state_manager = mock.Mock()
        self.mock_data_loader = mock.Mock()
        self.mock_text_extractor = mock.Mock()
        self.mock_embedding_generator = mock.Mock()
        self.mock_data_saver = mock.Mock()

        # FIX: Ensure 'save_batch' attribute exists on mock_data_saver
        # This addresses the AttributeError: Mock object has no attribute 'save_batch'
        self.mock_data_saver.save_batch = mock.MagicMock()

        # Initialize the pipeline with mocks
        self.pipeline = EmbeddingPipeline(
            config=self.mock_config,
            logger=self.mock_logger,
            state_manager=self.mock_state_manager,
            data_loader=self.mock_data_loader,
            text_extractor=self.mock_text_extractor,
            embedding_generator=self.mock_embedding_generator,
            data_saver=self.mock_data_saver,
        )

    def test_run_no_new_pages(self):
        """Test the run method when there are no new pages to process."""
        self.mock_state_manager.get_processed_urls.return_value = {
            "http://existing.com"
        }
        # Simulate no new data coming from the data loader
        self.mock_data_loader.stream_unprocessed_data.return_value = []

        # Run the pipeline
        results = list(self.pipeline.run())

        # Assertions
        self.assertIn("Initializing...", results)
        self.assertIn("Loading model and querying data...", results)
        self.assertIn("Already up to date.", results)

        self.mock_state_manager.get_processed_urls.assert_called_once()
        self.mock_data_loader.stream_unprocessed_data.assert_called_once_with(
            {"http://existing.com"}, self.mock_config.batch_size
        )
        self.mock_text_extractor.extract.assert_not_called()
        self.mock_embedding_generator.generate.assert_not_called()
        self.mock_data_saver.save_batch.assert_not_called()
        self.mock_logger.info.assert_called_with(
            "No new pages to process. The dataset is already up to date."
        )

    def test_run_with_data_processing(self):
        """Test the run method with multiple batches of data to process."""
        # Setup mock data for two batches
        df_batch_1 = pd.DataFrame(
            {
                "URL": ["http://page1.com", "http://page2.com"],
                "Content": ["<html>page1</html>", "<html>page2</html>"],
            }
        )
        df_batch_2 = pd.DataFrame(
            {"URL": ["http://page3.com"], "Content": ["<html>page3</html>"]}
        )

        self.mock_state_manager.get_processed_urls.return_value = set()
        # The data loader will yield two DataFrames (two batches)
        self.mock_data_loader.stream_unprocessed_data.return_value = [
            df_batch_1,
            df_batch_2,
        ]

        # FIX: Make these strings genuinely longer than 100 characters to pass the filter.
        # Count characters carefully or generate them to be very long.
        # Example for clarity: 'X' repeated 110 times for each.
        long_text_1 = (
            "X" * 110
            + "This is clean text for page 1. It must be sufficiently long to pass the filter in the pipeline, which requires more than one hundred characters. This string is now definitely long enough. Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        )
        long_text_2 = (
            "Y" * 110
            + "This is clean text for page 2. It must be sufficiently long to pass the filter in the pipeline, which requires more than one hundred characters. This string is now definitely long enough. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        )
        long_text_3 = (
            "Z" * 110
            + "This is clean text for page 3. It must be sufficiently long to pass the filter in the pipeline, which requires more than one hundred characters. This string is now definitely long enough. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
        )

        self.mock_text_extractor.extract.side_effect = [
            long_text_1,  # For page1.com
            long_text_2,  # For page2.com
            long_text_3,  # For page3.com
        ]

        # Mock embedding generator to return dummy embeddings for each batch
        # This will be called twice: once for the 2 items from batch 1, once for the 1 item from batch 2
        self.mock_embedding_generator.generate.side_effect = [
            [
                np.array([0.1, 0.2]),
                np.array([0.3, 0.4]),
            ],  # Embeddings for batch 1 (2 items)
            [np.array([0.5, 0.6])],  # Embeddings for batch 2 (1 item)
        ]

        # Run the pipeline and collect all yielded status messages
        results = list(self.pipeline.run())

        # Assertions for pipeline status messages
        self.assertIn("Processing Batch 1 (2 pages)...", results)
        # The test expects to see 'Processing Batch 2 (1 pages)...'
        self.assertIn("Processing Batch 2 (1 pages)...", results)
        self.assertIn("Finished", results)

        # Assertions for mock calls
        self.mock_state_manager.get_processed_urls.assert_called_once()
        self.mock_data_loader.stream_unprocessed_data.assert_called_once_with(
            set(), self.mock_config.batch_size
        )

        # The extract method is called once for each page (2 in batch 1 + 1 in batch 2 = 3 calls)
        self.assertEqual(self.mock_text_extractor.extract.call_count, 3)
        self.mock_text_extractor.extract.assert_has_calls(
            [
                mock.call("<html>page1</html>"),
                mock.call("<html>page2</html>"),
                mock.call("<html>page3</html>"),
            ]
        )

        # Embedding generator is called once per *filtered* batch
        self.assertEqual(self.mock_embedding_generator.generate.call_count, 2)
        self.mock_embedding_generator.generate.assert_has_calls(
            [
                mock.call([long_text_1, long_text_2]),  # Check with the long texts
                mock.call([long_text_3]),  # Check with the long text
            ]
        )

        # Data saver is called once per *processed* batch
        self.assertEqual(self.mock_data_saver.save_batch.call_count, 2)
        self.mock_data_saver.save_batch.assert_has_calls(
            [
                mock.call(mock.ANY, 1),  # Check the batch number
                mock.call(mock.ANY, 2),
            ]
        )
        # More precise check for save_batch arguments (optional but good practice)
        # Check first batch save
        args, _ = self.mock_data_saver.save_batch.call_args_list[0]
        self.assertIsInstance(args[0], pd.DataFrame)
        self.assertListEqual(
            args[0]["URL"].tolist(), ["http://page1.com", "http://page2.com"]
        )
        self.assertTrue(all(isinstance(e, list) for e in args[0]["Embedding"]))

        # Check second batch save
        args, _ = self.mock_data_saver.save_batch.call_args_list[1]
        self.assertIsInstance(args[0], pd.DataFrame)
        self.assertListEqual(args[0]["URL"].tolist(), ["http://page3.com"])
        self.assertTrue(all(isinstance(e, list) for e in args[0]["Embedding"]))

        self.mock_logger.info.assert_called_with(
            "All new batches processed successfully."
        )

    def test_run_batch_with_insufficient_text(self):
        """Test that batches with insufficient extracted text are skipped if some pass."""
        df_batch = pd.DataFrame(
            {
                "URL": ["http://shortcontent.com", "http://longcontent.com"],
                "Content": ["<html>short</html>", "<html>long enough</html>"],
            }
        )

        self.mock_state_manager.get_processed_urls.return_value = set()
        self.mock_data_loader.stream_unprocessed_data.return_value = [df_batch]

        # Mock text extractor: one short text, one long text (filter is > 100 chars)
        short_text = "short text"  # length 10
        long_text_passing_filter = "This is a much longer piece of text that definitely has more than one hundred characters, ensuring it passes the filter criterion set in the pipeline code. Yes, this is quite long now and passes the 100 character threshold. Testing completed."  # length > 100

        self.mock_text_extractor.extract.side_effect = [
            short_text,
            long_text_passing_filter,
        ]
        # Mock embedding generator will only be called for the long text
        self.mock_embedding_generator.generate.return_value = [np.array([0.7, 0.8])]

        # Run the pipeline
        results = list(self.pipeline.run())

        # Assertions
        self.assertIn("Processing Batch 1 (2 pages)...", results)
        self.assertIn(
            "Finished", results
        )  # Still finishes because the pipeline went through all batches

        self.assertEqual(
            self.mock_text_extractor.extract.call_count, 2
        )  # Called for both inputs
        self.mock_text_extractor.extract.assert_has_calls(
            [mock.call("<html>short</html>"), mock.call("<html>long enough</html>")]
        )

        # Embedding generator should only be called once, for the single valid text
        self.assertEqual(self.mock_embedding_generator.generate.call_count, 1)
        self.mock_embedding_generator.generate.assert_called_once_with(
            [long_text_passing_filter]
        )

        # Data saver should be called once, for the single valid item
        self.mock_data_saver.save_batch.assert_called_once()
        args, _ = self.mock_data_saver.save_batch.call_args_list[0]
        self.assertIsInstance(args[0], pd.DataFrame)
        self.assertEqual(len(args[0]), 1)  # Only one item should be saved
        self.assertListEqual(args[0]["URL"].tolist(), ["http://longcontent.com"])

    def test_run_batch_completely_insufficient_text(self):
        """Test that a batch entirely composed of insufficient text is skipped."""
        df_batch = pd.DataFrame(
            {
                "URL": ["http://short1.com", "http://short2.com"],
                "Content": ["<html>s1</html>", "<html>s2</html>"],
            }
        )

        self.mock_state_manager.get_processed_urls.return_value = set()
        self.mock_data_loader.stream_unprocessed_data.return_value = [df_batch]

        # Mock text extractor to return only short texts (all will be filtered out)
        self.mock_text_extractor.extract.side_effect = [
            "short text 1",  # length 12 <= 100
            "short text 2",  # length 12 <= 100
        ]

        # Run the pipeline
        results = list(self.pipeline.run())

        # Assertions
        self.assertIn("Processing Batch 1 (2 pages)...", results)
        self.assertIn(
            "Finished", results
        )  # Pipeline still reports finished after attempting all batches

        # Check that the logger recorded the "no pages" message
        info_calls = [
            call_args[0][0] for call_args in self.mock_logger.info.call_args_list
        ]
        self.assertIn(
            "Batch had no pages with sufficient text after cleaning.", info_calls
        )

        self.assertEqual(
            self.mock_text_extractor.extract.call_count, 2
        )  # extract is still called for both inputs
        self.mock_embedding_generator.generate.assert_not_called()  # Should not generate any embeddings
        self.mock_data_saver.save_batch.assert_not_called()  # Should not save any batch

    def test_run_exception_handling(self):
        """Test that exceptions are caught and reported."""
        # Make a mock method raise an exception
        self.mock_state_manager.get_processed_urls.side_effect = Exception(
            "Test Error during state manager call"
        )

        # Run the pipeline and capture output
        results = list(self.pipeline.run())

        # Assert that the error message is yielded and logger.exception is called
        self.assertIn("Error: Test Error during state manager call", results)
        self.mock_logger.exception.assert_called_once()
        self.mock_logger.exception.assert_called_with(
            "A critical pipeline error occurred: Test Error during state manager call"
        )
