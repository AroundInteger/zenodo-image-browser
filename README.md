# Zenodo Image Browser

A web application for browsing and analyzing experimental images from Zenodo datasets. This platform provides researchers with tools to explore, visualize, and analyze experimental data with AI-powered assistance.

## Project Vision

A centralized, web-based platform for researchers to explore, analyze, and benchmark experimental data (images, videos, rheology, etc.) with AI-powered assistance for:
- Data exploration and visualization
- Automated benchmarking and comparison to literature
- Generating analysis reports
- Guiding users through data experiments, even across disciplines

## Features

- Browse and search Zenodo datasets
- Preview images and videos
- Interactive data visualization
- AI-powered analysis and benchmarking
- Custom experiment building
- Automated report generation

## Key Features

### Data Access & Management
- Connect directly to Zenodo via API to fetch and index datasets
- Browse, search, and filter experiments by metadata
- Preview images/videos and download raw data

### Interactive Analysis
- Visualization tools for images, videos, phase diagrams, and time series
- Rheology/physics analysis with feature extraction and statistics
- Custom experiment builder for data analysis

### AI/ML Integration
- AI agents for analysis suggestions and benchmarking
- ML models for image/video classification and pattern recognition
- Automated report generation and natural language queries

### Collaboration & Reporting
- Export/share analysis reports (PDF, HTML, Markdown)
- Data annotation and comments
- Integration with literature databases

## Tech Stack

- Frontend: Streamlit
- Backend: Python (FastAPI)
- AI/ML: HuggingFace, scikit-learn, PyTorch
- Data Storage: Zenodo API

## Setup

1. Clone the repository:
```bash
git clone https://github.com/AroundInteger/zenodo-image-browser.git
cd zenodo-image-browser
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## Development

- `app.py`: Main Streamlit application
- `requirements.txt`: Project dependencies
- `src/`: Source code directory
  - `api/`: API integration with Zenodo
  - `models/`: AI/ML models
  - `utils/`: Utility functions
  - `visualization/`: Data visualization components

## Example User Stories

- "I want to upload a new experiment and see how its results compare to published phase diagrams."
- "Show me all experiments with phi > 0.5 and rate < 10, and plot their instability patterns."
- "Generate a report summarizing the main findings from the 2023 dataset."
- "What is the most similar published experiment to this new data?"

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
