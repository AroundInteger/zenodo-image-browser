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

## Development Workflow & Best Practices

This section outlines the recommended development practices for maintaining a clean, collaborative codebase.

### Git & GitHub Best Practices

#### Branching Strategy
- **Never work directly on `main`** - always use feature branches
- Create descriptive branch names:
  ```bash
  git checkout -b feature/zip-extraction
  git checkout -b fix/image-display-bug
  git checkout -b docs/update-readme
  ```

#### Commit Guidelines
- **Commit early and often** with small, focused changes
- Use [Conventional Commits](https://www.conventionalcommits.org/) format:
  ```bash
  git commit -m "feat: add ZIP file extraction support"
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

## New in 2025: ZIP Archive Support & Streamlit 1.40+

- The app now automatically extracts and displays images from ZIP files in Zenodo records. You can preview and analyze images inside ZIPs just like regular files.
- All image display now uses the new `use_container_width` parameter, fully compatible with Streamlit 1.40+.
- To use the latest features, ensure you have Streamlit 1.40 or newer installed in your environment.

### ZIP Workflow
- When you open a Zenodo record, any ZIP files are automatically scanned and their contents listed alongside top-level files.
- Selecting an image from inside a ZIP will extract and display it in memory, with full support for analysis tools.
