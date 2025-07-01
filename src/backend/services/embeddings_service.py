import pandas as pd
from tqdm import tqdm

from src.backend.config.embeddings_config import EmbeddingConfig
from src.backend.data.embedding_state_manager import EmbeddingStateManager
from src.backend.data.embeddings_loader import DataLoader
from src.backend.data.embeddings_saver import DataSaver
from src.backend.utils.text_processing import TextExtractor
from src.backend.utils.embedding_generation import (
    EmbeddingGenerator,
)  # Ensure this path is correct for EmbeddingGenerator
from src.shared.interfaces import ILogger


class EmbeddingPipeline:
    """Orchestrates the entire embedding generation process."""

    def __init__(
        self,
        config: EmbeddingConfig,
        logger: ILogger,
        state_manager: EmbeddingStateManager,
        data_loader: DataLoader,
        text_extractor: TextExtractor,
        embedding_generator: EmbeddingGenerator,
        data_saver: DataSaver,
    ):
        self.config = config
        self.logger = logger
        self.state_manager = state_manager
        self.data_loader = data_loader
        self.text_extractor = text_extractor
        self.embedding_generator = embedding_generator
        self.data_saver = data_saver

    # In src/backend/services/embeddings_service.py, inside the run method

    def run(self):
        """A generator that executes the pipeline and yields status updates."""
        try:
            yield "Initializing..."
            print("--- DEBUG: Pipeline Init ---")  # ADD THIS LINE
            processed_urls = self.state_manager.get_processed_urls()

            yield "Loading model and querying data..."
            print("--- DEBUG: Loading and Querying ---")  # ADD THIS LINE
            data_stream = self.data_loader.stream_unprocessed_data(
                processed_urls, self.config.batch_size
            )

            batch_num = 1
            processed_in_this_session = False
            for df_batch in data_stream:
                print(
                    f"--- DEBUG: STARTING BATCH. Current batch_num: {batch_num}, DataFrame size: {len(df_batch)} ---"
                )  # ADD THIS LINE
                processed_in_this_session = True
                status_msg = f"Processing Batch {batch_num} ({len(df_batch)} pages)..."
                self.logger.info(status_msg)
                yield status_msg

                # Extract Text (rest of your existing code)
                df_batch["clean_text"] = [
                    self.text_extractor.extract(html)
                    for html in tqdm(
                        df_batch["Content"], desc="Extracting Text", unit="docs"
                    )
                ]
                df_batch = df_batch[df_batch["clean_text"].str.len() > 100]

                if df_batch.empty:
                    self.logger.info(
                        "Batch had no pages with sufficient text after cleaning."
                    )
                    print(
                        f"--- DEBUG: Batch {batch_num} EMPTY after filter. Skipping increment. ---"
                    )  # ADD THIS LINE
                    continue  # If continues, batch_num is NOT incremented for this "empty" batch

                # Generate Embeddings (rest of your existing code)
                embeddings = self.embedding_generator.generate(
                    df_batch["clean_text"].tolist()
                )

                # Save Batch (rest of your existing code)
                output_df = pd.DataFrame(
                    {
                        "URL": df_batch["URL"],
                        "Embedding": [e.tolist() for e in embeddings],
                    }
                )
                self.data_saver.save_batch(output_df, batch_num)
                batch_num += 1  # <--- THIS IS THE LINE THAT MUST BE HERE AND EXECUTED
                print(
                    f"--- DEBUG: ENDING BATCH. batch_num incremented to: {batch_num} ---"
                )  # ADD THIS LINE

            print(
                f"--- DEBUG: Exited data_stream loop. Final batch_num: {batch_num} ---"
            )  # ADD THIS LINE
            if not processed_in_this_session:
                self.logger.info(
                    "No new pages to process. The dataset is already up to date."
                )
                yield "Already up to date."
            else:
                self.logger.info("All new batches processed successfully.")
                yield "Finished"

        except Exception as e:
            self.logger.exception(f"A critical pipeline error occurred: {e}")
            yield f"Error: {e}"
