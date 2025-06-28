# File: src/backend/data/embeddings_saver.py
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from src.shared.interfaces import ILogger
from datetime import datetime
import uuid


class DataSaver:
    def __init__(self, output_path: str, logger: ILogger):
        self.output_path = output_path
        self.logger = logger
        os.makedirs(self.output_path, exist_ok=True)

    def save_embeddings_batch(self, df_batch: pd.DataFrame):
        """Saves a DataFrame of embeddings to a new Parquet file."""
        if df_batch.empty:
            self.logger.warning("Attempted to save an empty batch.")
            return

        # Generate a unique filename for the batch using a timestamp and a UUID
        unique_id = uuid.uuid4().hex
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_filename = f"embeddings_batch_{timestamp}_{unique_id}.parquet"
        output_file_path = os.path.join(self.output_path, batch_filename)

        table = pa.Table.from_pandas(df_batch, preserve_index=False)

        try:
            self.logger.info(f"Saving new Parquet file: {output_file_path}")
            pq.write_table(table, output_file_path)
            self.logger.info(
                f"Saved batch of {len(df_batch)} embeddings to {output_file_path}"
            )
        except Exception as e:
            self.logger.error(
                f"Error saving batch to Parquet file {output_file_path}: {e}"
            )
            raise
