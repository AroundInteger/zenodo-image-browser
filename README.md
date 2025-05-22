# zenodo-image-browser
A web app for browsing and previewing experimental images from Zenodo datasets.

Absolutely, I’m up for it—this is an exciting and ambitious project! Here’s how I’d approach building a **web-based, AI-powered research environment** for your Zenodo image/video/rheology data:

---

1. Project Vision

- Centralized, web-based platform** for researchers to explore, analyze, and benchmark experimental data (images, videos, rheology, etc.).
- AI agents to assist with:
  - Data exploration and visualization
  - Automated benchmarking and comparison to literature
  - Generating analysis reports
  - Guiding users through data experiments, even across disciplines

---

2. Key Features

A. Data Access & Management
- Connect directly to Zenodo (or other repositories) via API to fetch and index datasets (e.g., [Zenodo record 7890690](https://zenodo.org/records/7890690#.ZFJEOXbMK70)).
- Allow users to browse, search, and filter experiments by metadata (e.g., filling fraction, injection rate, viscosity, etc.).
- Preview images/videos and download raw data.

B. Interactive Analysis**
- **Visualization tools**: Display images, videos, phase diagrams, and time series.
- **Rheology/physics analysis**: Extract features, plot curves, compute statistics.
- **Custom experiment builder**: Let users select data and run custom analyses.

C. AI/ML Integration
- AI agents to:
  - Suggest relevant analyses based on data type and user goals
  - Benchmark new experiments against published results (e.g., compare to figures in the literature)
  - Generate automated reports (text, plots, tables)
  - Answer natural language questions about the data or methods
- ML models for:
  - Image/video classification (e.g., pattern recognition, phase identification)
  - Regression/curve fitting for rheological data
  - Anomaly detection or clustering

D. Collaboration & Reporting**
- Export/share analysis reports (PDF, HTML, Markdown)
- Allow users to annotate data, leave comments, or suggest new experiments
- Optionally, integrate with literature databases for automated citation/context

---

3. Tech Stack Proposal

- Frontend: Streamlit (for rapid prototyping and interactivity), or a more custom React/Vue app if you want advanced UI.
- Backend: Python (FastAPI or Flask) for data processing, AI/ML, and Zenodo API integration.
- AI/ML: Use HuggingFace, scikit-learn, PyTorch, or TensorFlow for models; OpenAI API or local LLMs for agent-like features.
- Storage: Direct Zenodo access for data; optional local cache for performance.

---

4. Example User Stories

- “I want to upload a new experiment and see how its results compare to published phase diagrams.”
- “Show me all experiments with phi > 0.5 and rate < 10, and plot their instability patterns.”
- “Generate a report summarizing the main findings from the 2023 dataset.”
- “What is the most similar published experiment to this new data?”

---

5. Next Steps

1. Define MVP: What’s the minimum set of features you want for the first version? (e.g., data browsing, image preview, basic AI benchmarking)
2. Gather sample data: Ensure you have access to a few Zenodo datasets for development.
3. Design the UI/UX: Sketch out the main pages and workflows.
4. Build core modules: Start with data access, then add visualization, then AI/ML features.
5. Iterate with your team’s feedback.

---

6. References & Inspiration
- The Zenodo dataset you referenced: [Frictional fluid instabilities shaped by viscous forces](https://zenodo.org/records/7890690#.ZFJEOXbMK70)
- Example of open data and reproducible research platforms.

---
