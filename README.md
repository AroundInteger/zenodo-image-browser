# Zenodo Image Browser

A web application for browsing and analyzing experimental images from Zenodo datasets. This platform provides researchers with tools to explore, visualize, and analyze experimental data with AI-powered assistance.

## Features

- Browse and search Zenodo datasets
- Preview images and videos
- Interactive data visualization
- AI-powered analysis and benchmarking
- Custom experiment building
- Automated report generation

## Tech Stack

- Frontend: Streamlit
- Backend: Python (FastAPI)
- AI/ML: HuggingFace, scikit-learn
- Data Storage: Zenodo API

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/zenodo-image-browser.git
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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
