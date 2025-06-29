# How WebKnoGraph Works

## 1. Project Overview and Goals

WebKnoGraph is a project designed to revolutionize website internal linking by leveraging data processing techniques, vector embeddings, and graph-based link prediction algorithms. The primary goal is to create an intelligent solution that optimizes internal link structures, thereby enhancing SEO performance and user navigation. It aims to provide the first publicly available and transparent research for academic and industry purposes in end-to-end SEO and technical marketing.

The project is targeted at tech-savvy marketers and marketing engineers who have a strong understanding of data analytics and data-driven marketing strategies. It is particularly beneficial for mid to large-sized organizations managing a high volume of content (1,000+ pages) and looking to scale their internal linking efforts with precision and control.

## 2. Directory Structure

The project is organized into a modular structure:

```
WebKnoGraph/
├── assets/             # Project assets (images, logos)
├── data/               # Runtime output (crawled data, embeddings, models - typically not in repo)
│   ├── crawled_data_parquet/  # Raw HTML content from crawler
│   ├── url_embeddings/        # Vector embeddings for URLs
│   ├── link_graph_edges.csv   # Extracted internal links (edge list)
│   ├── url_analysis_results.csv # PageRank, HITS, folder depth per URL
│   └── prediction_model/      # Trained GNN model and metadata
├── notebooks/          # Jupyter notebooks acting as Gradio UIs for each module
│   ├── crawler_ui.ipynb
│   ├── embeddings_ui.ipynb
│   ├── link_crawler_ui.ipynb
│   ├── link_prediction_ui.ipynb
│   └── pagerank_ui.ipynb
├── src/                # Core source code
│   ├── backend/        # Backend logic
│   │   ├── config/     # Configuration for each module
│   │   ├── data/       # Data loading, saving, state management
│   │   ├── graph/      # Graph-specific algorithms (PageRank, HITS)
│   │   ├── models/     # Machine learning model definitions (GraphSAGE)
│   │   ├── services/   # Core business logic and orchestration
│   │   └── utils/      # Utility functions (HTTP, URL processing, text extraction)
│   └── shared/         # Components shared (e.g., logging, interfaces)
├── tests/              # Unit tests
│   └── backend/
│       └── services/
├── technical_report/   # Detailed technical documentation
│   └── WebKnoGraph_Technical_Report.pdf
├── .github/            # GitHub Actions workflows
├── LICENSE
├── README.md
├── requirements.txt
└── ... (other project files)
```

## 3. Data Flow and Processing Steps

WebKnoGraph processes website data through a series of interconnected modules, typically run in sequence using the Jupyter notebook UIs.

### 3.1. Web Crawling
-   **UI Notebook:** `notebooks/crawler_ui.ipynb`
-   **Service:** `src/backend/services/crawler_service.py`
-   **Configuration:** `src/backend/config/crawler_config.py`
-   **Process:**
    1.  Takes a starting URL and crawl parameters (e.g., depth, strategy like BFS/DFS).
    2.  Fetches web pages using HTTP requests (`src/backend/utils/http.py`).
    3.  Extracts text content from HTML (`src/backend/utils/text_processing.py`).
    4.  Filters URLs based on defined rules (`src/backend/utils/url.py`).
    5.  Saves the crawled content (HTML, text) into Parquet files in the `data/crawled_data_parquet/` directory, partitioned by crawl date.
    6.  Maintains crawl state in an SQLite database (`data/crawler_state.db`) to allow resuming crawls, managed by `src/backend/data/repositories.py`.

### 3.2. Embeddings Generation
-   **UI Notebook:** `notebooks/embeddings_ui.ipynb`
-   **Service:** `src/backend/services/embeddings_service.py`
-   **Configuration:** `src/backend/config/embeddings_config.py`
-   **Process:**
    1.  Reads the crawled text data from `data/crawled_data_parquet/`.
    2.  Uses a pre-trained sentence transformer model (e.g., from Hugging Face) to generate vector embeddings for the content of each URL (`src/backend/utils/embedding_generation.py`).
    3.  Saves the generated embeddings as Parquet files in `data/url_embeddings/`.
    4.  Manages the state of embedding generation using `src/backend/data/embedding_state_manager.py`, `src/backend/data/embeddings_loader.py`, and `src/backend/data/embeddings_saver.py`.

### 3.3. Link Graph Extraction
-   **UI Notebook:** `notebooks/link_crawler_ui.ipynb`
-   **Service:** `src/backend/services/link_crawler_service.py`
-   **Configuration:** `src/backend/config/link_crawler_config.py`
-   **Process:**
    1.  Re-crawls the website (or uses cached HTML if available from the initial crawl data) specifically to find internal hyperlinks.
    2.  Extracts `<a>` tags and their `href` attributes from the HTML of crawled pages.
    3.  Filters these links to include only internal links based on the domain (`src/backend/utils/link_url.py`).
    4.  Constructs an edge list representing the internal link graph (FROM URL, TO URL).
    5.  Saves this edge list as `data/link_graph_edges.csv`.
    6.  Uses `src/backend/data/link_graph_repository.py` for managing state and saving data.

### 3.4. Graph-Based Analysis (PageRank, HITS)
-   **UI Notebook:** `notebooks/pagerank_ui.ipynb`
-   **Service:** `src/backend/services/pagerank_service.py`
-   **Configuration:** `src/backend/config/pagerank_config.py`
-   **Graph Logic:** `src/backend/graph/analyzer.py`
-   **Process:**
    1.  Loads the link graph from `data/link_graph_edges.csv`.
    2.  Builds a NetworkX directed graph (`nx.DiGraph`) from the edge list.
    3.  Calculates PageRank scores for each URL (node) using `networkx.pagerank()`.
    4.  Calculates HITS (Hyperlink-Induced Topic Search) scores (Hubs and Authorities) for each URL using `networkx.hits()`.
    5.  Calculates folder depth for each URL (`src/backend/utils/url_processing.py`).
    6.  Saves the results (URL, PageRank, Hub Score, Authority Score, Folder Depth) into `data/url_analysis_results.csv`.

### 3.5. Link Prediction using GraphSAGE
-   **UI Notebook:** `notebooks/link_prediction_ui.ipynb`
-   **Services:**
    -   `src/backend/services/graph_training_service.py` (for model training)
    -   `src/backend/services/recommendation_engine.py` (for generating recommendations)
-   **Configuration:** `src/backend/config/link_prediction_config.py`
-   **Model Definition:** `src/backend/models/graph_models.py` (GraphSAGEModel)
-   **Data Handling:** `src/backend/data/graph_dataloader.py`, `src/backend/data/graph_processor.py`
-   **Process:**
    1.  **Data Preparation:**
        -   Loads the link graph from `data/link_graph_edges.csv`.
        -   Loads the URL embeddings from `data/url_embeddings/`.
        -   Creates a PyTorch Geometric `Data` object. Nodes are URLs, node features are their embeddings, and edges are the internal links.
    2.  **Model Training (`LinkPredictionTrainer`):**
        -   Initializes a `GraphSAGEModel` (a 2-layer GraphSAGE network using `torch_geometric.nn.SAGEConv`).
        -   Trains the model to predict links. This involves:
            -   Generating negative samples (pairs of nodes that are not linked).
            -   Using the GraphSAGE model to get embeddings for all nodes based on graph structure and initial features.
            -   Predicting the existence of an edge between two nodes by taking the dot product of their GraphSAGE-generated embeddings.
            -   Using Binary Cross-Entropy with Logits loss (`nn.BCEWithLogitsLoss`) to compare predictions against actual links (positive samples) and non-links (negative samples).
            -   Optimizing model parameters using Adam optimizer.
        -   Saves the trained model (`graphsage_link_predictor.pth`), final node embeddings (`final_node_embeddings.pt`), edge index (`edge_index.pt`), and metadata (`model_metadata.json`) to `data/prediction_model/`.
    3.  **Recommendation Generation (`RecommendationEngine`):**
        -   Loads the trained model and associated data.
        -   For a given source URL, it uses the trained GraphSAGE model to predict potential new links to other URLs on the site that are not currently linked from the source URL.
        -   Ranks these potential links by their predicted scores and presents them as recommendations.

## 4. Key Technologies and Libraries Used

-   **Python:** Core programming language.
-   **Jupyter Notebooks & Gradio:** For creating interactive UIs for each module.
-   **Pandas:** For data manipulation and analysis (e.g., handling CSVs, DataFrames).
-   **Parquet (pyarrow):** For efficient storage and retrieval of crawled data and embeddings.
-   **SQLite (sqlite3):** For managing crawler state.
-   **Requests & Beautiful Soup:** For web crawling and HTML parsing (implicitly, likely via `src/backend/utils/http.py` and `src/backend/utils/text_processing.py`).
-   **Sentence Transformers (Hugging Face):** For generating text embeddings.
-   **PyTorch:** As the underlying framework for PyTorch Geometric.
-   **PyTorch Geometric (PyG):**
    -   Used for building and training the GraphSAGE model for link prediction.
    -   Key components: `torch_geometric.nn.SAGEConv`, `torch_geometric.data.Data`.
-   **NetworkX:**
    -   Used for graph analysis tasks like PageRank and HITS.
    -   Key components: `nx.DiGraph`, `nx.pagerank()`, `nx.hits()`.
-   **Scikit-learn:** Likely used for utility functions or potentially other ML tasks not detailed here (though not explicitly seen in the core graph logic examined).
-   **Ngrok:** Used in the notebooks to expose Gradio UIs publicly.

**Note on StellarGraph and Barabási-Albert (BA) model:**
Based on the codebase review:
-   **StellarGraph** is **not** directly used. The project relies on PyTorch Geometric for graph neural network implementations.
-   **NetworkX's Barabási-Albert (BA) model** for graph generation is **not** directly used. The graph is constructed from actual crawled website links.

## 5. How to Run the Project

The project is designed to be run module by module using the Jupyter notebooks in Google Colab, leveraging Google Drive for data persistence.

1.  **Prerequisites:**
    *   Google Account (for Colab and Drive).
    *   Python 3.8+.

2.  **Setup:**
    *   Clone or download the `WebKnoGraph` repository.
    *   Upload the `WebKnoGraph` folder to your Google Drive (e.g., into `My Drive/`).
    *   Ensure the project path in Colab notebooks is correctly set (default: `/content/drive/My Drive/WebKnoGraph/`).

3.  **Running Modules (Notebooks):**
    *   Open a notebook (e.g., `notebooks/crawler_ui.ipynb`) in Google Colab.
    *   **Crucially**: Perform `Runtime -> Disconnect and delete runtime` for a clean start or if code has changed.
    *   Run the first cell to mount Google Drive and install dependencies (uncomment `!pip install` lines if needed, or run `!pip install -r requirements.txt`).
    *   Go to `Runtime -> Run all cells`.
    *   A Gradio UI link will appear in the output of the last cell.

4.  **Recommended Order of Execution:**
    1.  **Content Crawler:** `notebooks/crawler_ui.ipynb`
        *   Output: `data/crawled_data_parquet/`
    2.  **Embeddings Pipeline:** `notebooks/embeddings_ui.ipynb`
        *   Requires: Output from Content Crawler.
        *   Output: `data/url_embeddings/`
    3.  **Link Graph Extractor:** `notebooks/link_crawler_ui.ipynb`
        *   Output: `data/link_graph_edges.csv`
    4.  **PageRank & HITS Analysis:** `notebooks/pagerank_ui.ipynb`
        *   Requires: Output from Link Graph Extractor.
        *   Output: `data/url_analysis_results.csv`
    5.  **GNN Link Prediction & Recommendation Engine:** `notebooks/link_prediction_ui.ipynb`
        *   Requires: Outputs from Link Graph Extractor and Embeddings Pipeline.
        *   Output: `data/prediction_model/` (trained model and artifacts)

For detailed setup, troubleshooting, and explanations of each module's UI, refer to the main `README.md` file.

## 6. Starting a Fresh Crawl

To begin analysis for a new website or to restart, delete the entire `data/` folder. This ensures no residual data from previous runs interferes with the new session.

This document provides an in-depth understanding of WebKnoGraph's architecture, data flow, and core functionalities. For more specific details, refer to the source code and the `technical_report/WebKnoGraph_Technical_Report.pdf`.
