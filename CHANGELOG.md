# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.1] - 2025-06-28
### Changed
- Reduced save intervals in `embeddings_ui.ipynb` to prevent browser crashes during prolonged usage
- Updated App UI images in `README.md` to match the current interface

## [0.1.0] - 2025-06-28
### Added
- Enabled embeddings saving in `embeddings_ui.ipynb`, now aligned with batch size for consistent processing

### Changed
- Changed the config for the embeddings batch size to have value of 1
- Reorganized and cleaned up code structure in `embeddings_ui.ipynb`
- Replaced slider with input number field in `embeddings_ui.ipynb`, using a step value of 1

### Fixed
- Resolved non-functional slider in `embeddings_ui.ipynb`

## [0.1.0] - 2025-06-27
### Added
- Late release of WebKnoGraph with modular pipeline for:
  - Content crawler, embedding generator, link‑graph extractor
  - GraphSAGE link‑prediction model and PageRank/HITS utilities
  - Jupyter notebooks acting as Gradio UIs for each module
  - Unit tests for backend services (crawler, embeddings, graph training, pagerank)
  - GitHub Actions testing workflow
