GROQ_MODEL='openai/gpt-oss-20b'
NLP_MODEL="ProsusAI/finbert"
SENT_TRANSFORMER = "all-MiniLM-L6-v2"
# SENT_TRANSFORMER="paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER="cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Configuration (free-tier Groq models only) ---
EVAL_MODEL = "groq/qwen/qwen3.8-27b"     # judge
GEN_MODEL  = "groq/openai/gpt-oss-20b"     # RAG answer generation
GT_MODEL   = "groq/qwen/qwen3.8-27b"        # ground-truth extraction