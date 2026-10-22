# Finalyze

Finalyze is a Retrieval-Augmented Generation (RAG) application for analyzing earnings calls and financial research text. It combines transcript data, sentiment analysis, topic modeling, and hybrid retrieval to help users ask natural-language questions about company performance, supply-chain issues, AI strategy, and other earnings-related themes.

## Project overview

This project includes:

- A Streamlit dashboard for exploring earnings data and asking questions
- Hybrid retrieval using FAISS + BM25 with reranking
- Sentiment and topic analysis for earnings call content
- Optional live market-data augmentation for relevant company questions
- Data preparation and indexing scripts for structured financial text analysis

## Tech stack

- Python
- Streamlit
- Pandas
- Hugging Face sentence-transformers and reranking models
- LangChain / FAISS / BM25 retrieval
- Groq LLMs for generation and evaluation

## Repository structure

- `app.py` — Streamlit application UI
- `config.py` — model and configuration settings
- `retrieve.py` — retrieval logic and reranking
- `generate.py` — answer generation flow
- `load_data.py` — loading raw data
- `preprocess.py` — text preparation and cleanup
- `ml_sentiment.py` — sentiment analysis
- `ml_topics.py` — topic extraction
- `build_index.py` — vector / BM25 index creation
- `data/` — dataset and generated indexes
- `requirements.txt` — Python dependencies

## Getting started

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
. .venv/bin/activate    # Linux/macOS
# or
.venv\Scripts\activate # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

This project expects a Groq API key.

Create a `.env` file in the project root with:

```bash
GROQ_API_KEY=your_api_key_here
```

You can also export it directly in your shell:

```bash
export GROQ_API_KEY="your_api_key_here"
```

### 4. Build the dataset and indexes

If the processed data is not already present, run the pipeline:

```bash
python load_data.py
python preprocess.py
python ml_sentiment.py
python ml_topics.py
python build_index.py
```

### 5. Run the app

```bash
streamlit run app.py
```

Then open the local Streamlit URL in your browser.

## Usage

Use the dashboard to:

- filter by company, year, topic, and sentiment range
- ask questions in natural language about earnings discussions
- inspect retrieved transcript chunks and generated answers
- optionally include live market data for company fundamentals

## Notes

- The project is tuned for earnings-report-style data and financial language.
- Many model operations are cloud-backed through Groq and Hugging Face.
- Large model downloads and indexing may take time on first run.

## License

This project does not include a license file. If you plan to distribute or reuse it, confirm the licensing requirements for any included data, models, and third-party dependencies before publishing.
