# File: src/backend/services/recommendation_engine.py
import torch
import json
import pandas as pd
from src.shared.interfaces import ILogger
from src.backend.config.link_prediction_config import LinkPredictionConfig
from src.backend.models.graph_models import GraphSAGEModel
from src.backend.utils.url_processing import URLProcessor


class RecommendationEngine:
    """Loads trained artifacts and provides link recommendations using a Top-K strategy."""

    def __init__(
        self, config: LinkPredictionConfig, logger: ILogger, url_processor: URLProcessor
    ):
        self.config = config
        self.logger = logger
        self.url_processor = url_processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.node_embeddings = None
        self.url_to_idx = None
        self.idx_to_url = None
        self.existing_edges = None

    def load_artifacts(self):
        """Loads the trained model, embeddings, and mappings into memory."""
        if self.model is not None:
            return True

        self.logger.info("Loading trained artifacts for recommendations...")
        try:
            with open(self.config.node_mapping_path, "r") as f:
                model_metadata = json.load(f)

            self.url_to_idx = model_metadata["url_to_idx"]
            in_channels = model_metadata["in_channels"]
            hidden_channels = model_metadata["hidden_channels"]
            out_channels = model_metadata["out_channels"]

            self.idx_to_url = {v: k for k, v in self.url_to_idx.items()}

            self.node_embeddings = torch.load(self.config.node_embeddings_path).to(
                self.device
            )
            edge_index = torch.load(self.config.edge_index_path)
            self.existing_edges = set(
                zip(edge_index[0].tolist(), edge_index[1].tolist())
            )

            self.model = GraphSAGEModel(in_channels, hidden_channels, out_channels)
            self.model.load_state_dict(torch.load(self.config.model_state_path))
            self.model.to(self.device)
            self.model.eval()

            self.logger.info("Artifacts loaded successfully.")
            return True
        except FileNotFoundError:
            self.logger.error(
                "Could not find trained model artifacts. Please run the training pipeline first."
            )
            return False
        except Exception as e:
            self.logger.error(f"An error occurred while loading artifacts: {e}")
            raise

    def get_recommendations(
        self,
        source_url: str,
        top_n: int = 20,
        min_folder_depth: int = 0,
        max_folder_depth: int = 10,
        folder_path_filter: str = None,
    ):
        if not self.load_artifacts():
            return (
                None,
                "Error: Trained model artifacts not found. Please run the training pipeline first.",
            )
        if source_url not in self.url_to_idx:
            return (
                None,
                f"Error: Source URL '{source_url}' not found in the graph's training data.",
            )

        source_idx = self.url_to_idx[source_url]
        num_nodes = len(self.url_to_idx)

        # 1. Generate scores for all possible links from the source node
        candidate_dest_indices = torch.arange(num_nodes, device=self.device)
        candidate_source_indices = torch.full_like(
            candidate_dest_indices, fill_value=source_idx
        )
        candidate_edge_index = torch.stack(
            [candidate_source_indices, candidate_dest_indices]
        )

        with torch.no_grad():
            scores = self.model.predict_link(self.node_embeddings, candidate_edge_index)

        # 2. Create a DataFrame from all possible candidates
        all_candidates_df = pd.DataFrame(
            {
                "DEST_IDX": candidate_dest_indices.cpu().numpy(),
                "SCORE": torch.sigmoid(scores).cpu().numpy(),
            }
        )

        # 3. Add URL and FOLDER_DEPTH columns
        # Use .get() with a default value to handle missing keys and prevent KeyError
        all_candidates_df["RECOMMENDED_URL"] = all_candidates_df["DEST_IDX"].apply(
            lambda idx: self.idx_to_url.get(idx, None)
        )

        # Drop rows with invalid URLs (where index was not found in mapping)
        all_candidates_df.dropna(subset=["RECOMMENDED_URL"], inplace=True)

        all_candidates_df["FOLDER_DEPTH"] = all_candidates_df["RECOMMENDED_URL"].apply(
            lambda url: self.url_processor.get_folder_depth(url)
        )

        # 4. Filter the DataFrame based on all criteria
        filtered_df = all_candidates_df.copy()

        # Filter out self-links
        filtered_df = filtered_df[filtered_df["DEST_IDX"] != source_idx]

        # Filter out existing links
        # Create a tuple column for easy set membership check
        filtered_df["SOURCE_IDX"] = source_idx
        filtered_df["EDGE_TUPLE"] = list(
            zip(filtered_df["SOURCE_IDX"], filtered_df["DEST_IDX"])
        )
        filtered_df = filtered_df[~filtered_df["EDGE_TUPLE"].isin(self.existing_edges)]

        # Apply the folder depth filter
        filtered_df = filtered_df[
            (filtered_df["FOLDER_DEPTH"] >= min_folder_depth)
            & (filtered_df["FOLDER_DEPTH"] <= max_folder_depth)
        ]

        # Apply the folder path filter if provided
        if folder_path_filter:
            self.logger.info(f"Applying folder path filter: {folder_path_filter}")
            filtered_df = filtered_df[
                filtered_df["RECOMMENDED_URL"].str.startswith(folder_path_filter)
            ]

        # 5. Sort the filtered DataFrame by score and take the top N
        final_recommendations_df = filtered_df.sort_values(
            by="SCORE", ascending=False
        ).head(top_n)

        # 6. Select the final columns and return
        final_recommendations_df = final_recommendations_df[
            ["RECOMMENDED_URL", "SCORE", "FOLDER_DEPTH"]
        ]

        if final_recommendations_df.empty:
            return (
                None,
                "No recommendations found matching the criteria (filters, existing links, etc.). Try adjusting filters or source URL.",
            )

        return final_recommendations_df, None
