# Mental Health Counselor Assistant

A Streamlit-based web application that provides AI-powered mental health counseling assistance using RAG (Retrieval-Augmented Generation) technology.

## Features

- **AI-Powered Counseling**: Uses OpenAI's GPT models to provide empathetic and supportive responses
- **RAG System**: Retrieval-Augmented Generation for contextually relevant responses
- **Streamlit Interface**: User-friendly web interface for easy interaction
- **Privacy-Focused**: Local data processing with secure API key management

## Project Structure

```
mental-health-counselor-assistant/
├── streamlit_app.py          # Main Streamlit application
├── streamlit_app_wokey.py    # Alternative version without API key
├── test_rag.py              # RAG system testing script
├── requirements.txt          # Python dependencies
├── docs/                    # Project documentation
│   └── project_charter.md   # Project charter and specifications
├── notebooks/               # Jupyter notebooks for development
│   ├── 01_data_exploration.ipynb
│   ├── 02_build_faiss_index.ipynb
│   └── 02_rag_system.ipynb
└── data/                    # Data files (excluded from git)
    ├── contexts.pkl
    ├── faiss_index/
    ├── response_vectors.pkl
    ├── responses.pkl
    └── vectorizer.pkl
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RozMolaei/MentalHealthPoC.git
cd MentalHealthPoC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
   - Create a file named `API_Key.txt` in the project root
   - Add your OpenAI API key to the file

## Usage

### Running the Main Application

```bash
streamlit run streamlit_app.py
```

### Running without API Key

```bash
streamlit run streamlit_app_wokey.py
```

## Configuration

- **API Key**: Place your OpenAI API key in `API_Key.txt`
- **Model**: Default model is `gpt-3.5-turbo` (configurable in the app)
- **Temperature**: Adjustable response creativity (0.0-1.0)

## Features

### Main Application (`streamlit_app.py`)
- Full-featured counseling assistant
- Requires OpenAI API key
- Advanced RAG capabilities
- Session management

### Alternative Version (`streamlit_app_wokey.py`)
- Simplified version without API key requirement
- Basic counseling responses
- Suitable for testing and demonstration

## Development

### Jupyter Notebooks
- `01_data_exploration.ipynb`: Initial data analysis and exploration
- `02_build_faiss_index.ipynb`: Building the FAISS index for RAG
- `02_rag_system.ipynb`: RAG system implementation and testing

### Testing
```bash
python test_rag.py
```

## Security Notes

- API keys are stored locally and excluded from version control
- Sensitive data files are excluded via `.gitignore`
- Virtual environment files are not committed to the repository

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with OpenAI's usage policies and local regulations regarding mental health services.

## Disclaimer

This application is designed for educational and research purposes only. It is not a substitute for professional mental health care. If you are experiencing a mental health crisis, please contact a licensed mental health professional or emergency services. 