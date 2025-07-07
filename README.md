# Zenodo Image Browser

A web application for browsing and analyzing experimental images from Zenodo datasets. This platform provides researchers with tools to explore, visualize, and analyze experimental data with AI-powered assistance.

## Project Vision

A centralized, web-based platform for researchers to explore, analyze, and benchmark experimental data (images, videos, rheology, etc.) with AI-powered assistance for:
- Data exploration and visualization
- Automated benchmarking and comparison to literature
- Generating analysis reports
- Guiding users through data experiments, even across disciplines

## 🚀 Latest Features (2025)

### 🤖 AI-Powered Analysis
- **Dual AI Support**: Choose between OpenAI GPT (cloud) and Local Ollama (free, offline)
- **Smart Model Management**: Automatic model detection and one-click installation
- **Scientific Context**: AI understands your dataset structure and provides intelligent analysis
- **Natural Language Queries**: Ask questions about your data in plain English

### 📦 Advanced ZIP Archive Support
- **Automatic Extraction**: ZIP files are automatically scanned and contents listed
- **In-Memory Processing**: Images extracted on-demand without downloading
- **Nested ZIP Support**: Handles complex archive structures with multiple levels
- **Smart File Management**: Context-aware download options for files inside ZIPs

### 🖼️ Enhanced Image Processing
- **Multi-Format Support**: PNG, JPG, TIFF, and other common scientific image formats
- **Real-time Analysis**: Edge detection, color analysis, and pattern recognition
- **Interactive Tools**: Image gallery, filters, and statistical analysis
- **Large Dataset Optimization**: Efficient handling of datasets with thousands of images

## Features

- Browse and search Zenodo datasets
- Preview images and videos
- Interactive data visualization
- AI-powered analysis and benchmarking
- Custom experiment building
- Automated report generation
- **NEW**: Local AI models for offline analysis
- **NEW**: Automatic ZIP archive extraction and browsing

## Key Features

### Data Access & Management
- Connect directly to Zenodo via API to fetch and index datasets
- Browse, search, and filter experiments by metadata
- Preview images/videos and download raw data
- **NEW**: Automatic ZIP archive extraction and content browsing
- **NEW**: Support for nested ZIP files and complex archive structures

### Interactive Analysis
- Visualization tools for images, videos, phase diagrams, and time series
- Rheology/physics analysis with feature extraction and statistics
- Custom experiment builder for data analysis
- **NEW**: Image gallery with filtering and search capabilities
- **NEW**: Real-time edge detection and color analysis

### AI/ML Integration
- AI agents for analysis suggestions and benchmarking
- ML models for image/video classification and pattern recognition
- Automated report generation and natural language queries
- **NEW**: Local Ollama integration for offline AI analysis
- **NEW**: OpenAI GPT integration for cloud-based AI assistance
- **NEW**: Smart model management and automatic installation

### Collaboration & Reporting
- Export/share analysis reports (PDF, HTML, Markdown)
- Data annotation and comments
- Integration with literature databases

## Tech Stack

- Frontend: Streamlit 1.40+
- Backend: Python (FastAPI)
- AI/ML: OpenAI GPT, Ollama (Local), HuggingFace, scikit-learn, PyTorch
- Data Storage: Zenodo API
- **NEW**: ZIP processing with zipfile and io libraries

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

4. **Optional**: Install Ollama for local AI support:
```bash
# macOS
brew install ollama
brew services start ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
sudo systemctl start ollama

# Windows
# Download from https://ollama.ai/download
```

5. Run the application:
```bash
streamlit run app.py
```

## AI Assistant Setup

### OpenAI GPT (Cloud-based)
1. Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Enter the key in the sidebar when using the Analysis section
3. Start asking questions about your data!

### Local Ollama (Free, Offline)
1. Install Ollama (see setup step 4 above)
2. The app will automatically detect available models
3. Click "Install" to download your preferred model (e.g., llama2:7b)
4. Start using local AI analysis - no internet required!

## Example User Stories

- "I want to upload a new experiment and see how its results compare to published phase diagrams."
- "Show me all experiments with phi > 0.5 and rate < 10, and plot their instability patterns."
- "Generate a report summarizing the main findings from the 2023 dataset."
- "What is the most similar published experiment to this new data?"
- **NEW**: "Extract and analyze all images from this ZIP archive of experimental data."
- **NEW**: "What types of scientific images are in this dataset and what analysis would be useful?"

## Development

- `app.py`: Main Streamlit application
- `requirements.txt`: Project dependencies
- `src/`: Source code directory
  - `api/`: API integration with Zenodo
  - `models/`: AI/ML models
  - `utils/`: Utility functions
  - `visualization/`: Data visualization components
  - `analysis/`: Image analysis tools
  - `components/`: UI components

## Recent Updates

### v1.2.0 - AI Integration & ZIP Support
- ✅ Added OpenAI GPT integration for cloud-based AI analysis
- ✅ Added Ollama integration for local, offline AI analysis
- ✅ Implemented automatic ZIP archive extraction and browsing
- ✅ Enhanced image processing with multi-format support
- ✅ Added smart model management and installation
- ✅ Improved error handling and user feedback
- ✅ Updated to Streamlit 1.40+ compatibility

### v1.1.0 - Core Features
- ✅ Basic Zenodo API integration
- ✅ Image browsing and preview
- ✅ Interactive analysis tools
- ✅ Data visualization components

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Development Workflow & Best Practices

This section outlines the recommended development practices for maintaining a clean, collaborative codebase.

### Git & GitHub Best Practices

#### Branching Strategy
- **Never work directly on `main`** - always use feature branches
- Create descriptive branch names:
  ```bash
  git checkout -b feature/zip-extraction
  git checkout -b feature/ai-integration
  git checkout -b fix/image-display-bug
  git checkout -b docs/update-readme
  ```

#### Commit Guidelines
- **Commit early and often** with small, focused changes
- Use [Conventional Commits](https://www.conventionalcommits.org/) format:
  ```bash
  git commit -m "feat: add ZIP file extraction support"
  git commit -m "feat: integrate OpenAI and Ollama AI assistants"
  git commit -m "fix: resolve image display compatibility issues"
  git commit -m "docs: update README with new features"
  git commit -m "refactor: improve edge detection performance"
  ```

#### Pull Request Workflow
1. **Create a feature branch** from `main`
2. **Make your changes** with clear, focused commits
3. **Create a Pull Request** with descriptive title and description
4. **Review your own work** before requesting review
5. **Merge only when ready** - keep `main` deployable

#### Commit Message Types
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

### Solo Development Workflow
Even when working alone, follow these practices:
- Use feature branches for all changes
- Create pull requests for significant features
- Write clear commit messages
- Keep `main` stable and deployable

### Team Development Workflow
When collaborating:
- **Pull latest changes** regularly: `git pull origin main`
- **Resolve conflicts early** to avoid merge hell
- **Use issues and project boards** for tracking
- **Review each other's code** before merging
- **Communicate changes** in PR descriptions

### Code Quality Guidelines
- **Write self-documenting code** with clear variable names
- **Add comments** for complex logic
- **Follow PEP 8** for Python code style
- **Test your changes** before committing
- **Update documentation** when adding features

### File Organization
- Keep related files together in appropriate directories
- Use descriptive file and function names
- Maintain a clean project structure
- Update `requirements.txt` when adding dependencies

### Before Committing
- [ ] Code runs without errors
- [ ] All tests pass (if applicable)
- [ ] Documentation is updated
- [ ] Commit message follows conventions
- [ ] No sensitive data is included

### Emergency Fixes
For urgent fixes that need immediate deployment:
1. Create a hotfix branch: `git checkout -b hotfix/critical-bug`
2. Make minimal, focused changes
3. Test thoroughly
4. Create PR and merge quickly
5. Tag the release: `git tag v1.0.1`

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
