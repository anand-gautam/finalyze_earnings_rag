# app.py
"""
Streamlit UI for Earnings RAG.
Run: streamlit run app.py --server.port 8501
"""
import json
import os
import logging
import sys


from config import GROQ_MODEL

logging.getLogger("streamlit").setLevel(logging.ERROR)

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency may be unavailable
    pd = None

try:
    import plotly.express as px
except Exception:  # pragma: no cover - optional dependency may be unavailable
    px = None

try:
    import streamlit as st
except Exception:  # pragma: no cover - the app only runs in a Streamlit environment
    st = None

try:
    from retrieve import EarningsRetriever
    from generate import generate_answer, generate_answer_with_tools, generate_answer_react, needs_live_data, _extract_symbol
except Exception:  # pragma: no cover - optional dependencies may be missing
    EarningsRetriever = None

    def generate_answer(query, context):
        return "LLM generation is unavailable because dependencies are missing."

    def generate_answer_with_tools(query, context, max_tool_rounds=3, force_tool=False, allowed_symbols=None):
        return "Tool-calling generation is unavailable.", []

    def needs_live_data(query):
        return False

    def _extract_symbol(query, allowed_symbols=None):
        return None


if st is not None:
    st.set_page_config(page_title="Finalyze", page_icon="📊", layout="wide")
    st.markdown(
        """<style>
            .main-header{font-size:2.5rem;font-weight:700;background:linear-gradient(90deg,#6366f1,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
            div[data-testid="stMetric"]{background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid #333;border-radius:12px;padding:1rem}
        </style>""",
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def load_chunks():
        if pd is None:
            return pd.DataFrame()

        chunk_paths = [
            "data/index/chunks.parquet",
            "data/index/chunks.csv",
        ]
        for path in chunk_paths:
            if os.path.exists(path):
                if path.endswith(".parquet"):
                    return pd.read_parquet(path)
                return pd.read_csv(path)
        return pd.DataFrame()

    @st.cache_resource
    def get_retriever():
        if EarningsRetriever is None:
            return None
        return EarningsRetriever()

    if not any(os.path.exists(path) for path in ["data/index/chunks.parquet", "data/index/chunks.csv"]):
        st.error(
            "Index not found. Run the pipeline first:\n\n"
            "```\n"
            "python ingestion/load_data.py\n"
            "python ingestion/preprocess.py\n"
            "python ingestion/ml_sentiment.py\n"
            "python ingestion/ml_topics.py\n"
            "python ingestion/build_index.py\n"
            "```"
        )
        st.stop()

    chunks = load_chunks()

    if chunks.empty:
        st.error("The index file is present but could not be loaded. Check the data/index contents.")
        st.stop()

    required_cols = {"symbol", "year", "quarter", "sentiment", "topic_name", "text"}
    missing = sorted(required_cols - set(chunks.columns))
    if missing:
        st.error(f"Index format is invalid. Missing columns: {', '.join(missing)}")
        st.stop()

    chunks = chunks.copy()
    for col in ["symbol", "company_name", "topic_name", "sentiment_label", "date"]:
        if col in chunks.columns:
            chunks[col] = chunks[col].fillna("N/A").astype(str)
    if "sentiment" in chunks.columns:
        chunks["sentiment"] = pd.to_numeric(chunks["sentiment"], errors="coerce")
    if "year" in chunks.columns:
        chunks["year"] = pd.to_numeric(chunks["year"], errors="coerce")
    if "quarter" in chunks.columns:
        chunks["quarter"] = pd.to_numeric(chunks["quarter"], errors="coerce")
    if "text" in chunks.columns:
        chunks["text"] = chunks["text"].fillna("").astype(str)

    st.sidebar.title("⚙️ Filters")
    symbols = sorted(chunks["symbol"].dropna().astype(str).unique())
    years = sorted(pd.to_numeric(chunks["year"], errors="coerce").dropna().astype(int).unique())
    topic_list = chunks[["topic_id", "topic_name"]].dropna(subset=["topic_name"]).drop_duplicates()

    sel_symbol = st.sidebar.multiselect("Company", symbols, default=[])
    sel_year = st.sidebar.selectbox("Year", ["All"] + [str(y) for y in years])
    sel_topic = st.sidebar.selectbox("Topic", ["All"] + topic_list["topic_name"].astype(str).tolist())
    sent_range = st.sidebar.slider("Sentiment Range", -1.0, 1.0, (-1.0, 1.0), 0.1)

    mask = pd.Series(True, index=chunks.index)
    if sel_symbol:
        mask &= chunks["symbol"].astype(str).isin(sel_symbol)
    if sel_year != "All":
        mask &= pd.to_numeric(chunks["year"], errors="coerce") == int(sel_year)
    if sel_topic != "All":
        mask &= chunks["topic_name"].astype(str) == str(sel_topic)
    mask &= pd.to_numeric(chunks["sentiment"], errors="coerce").between(sent_range[0], sent_range[1])
    filtered = chunks.loc[mask].copy()
    filtered = filtered.dropna(subset=["sentiment"]).copy()
    st.sidebar.markdown(f"**{len(filtered):,}** chunks match")

    st.markdown('<h1 class="main-header">📊 Finalyze</h1>', unsafe_allow_html=True)
    st.caption("Earnings RAG application to analyze and query financial earnings reports")

    def render_live_data_cards(tool_log):
        if not tool_log:
            return

        st.markdown("**Live market data:**")
        for tl in tool_log:
            payload = {}
            try:
                payload = json.loads(tl["result"])
            except Exception:
                payload = {}

            if not isinstance(payload, dict):
                st.write(str(payload)[:1500])
                continue

            if payload.get("error"):
                st.warning(payload["error"])
                continue

            symbol = payload.get("symbol", "N/A")
            current_price = payload.get("current_price")
            market_cap = payload.get("market_cap")
            trailing_pe = payload.get("trailing_pe")
            forward_pe = payload.get("forward_pe")
            dividend_yield = payload.get("dividend_yield_pct")
            revenue_ttm = payload.get("revenue_ttm")
            beta = payload.get("beta")

            st.caption(f"Source: {symbol} | {tl['tool']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${current_price:,.2f}" if current_price is not None else "N/A")
            c2.metric("Mkt Cap", f"${market_cap:,.0f}" if market_cap is not None else "N/A")
            c3.metric("P/E", f"{trailing_pe:.2f}" if trailing_pe is not None else "N/A")
            c4.metric("Div Yield", f"{dividend_yield:.2f}%" if dividend_yield is not None else "N/A")

            c5, c6, c7 = st.columns(3)
            c5.metric("Revenue (TTM)", f"${revenue_ttm:,.0f}" if revenue_ttm is not None else "N/A")
            c6.metric("Forward P/E", f"{forward_pe:.2f}" if forward_pe is not None else "N/A")
            c7.metric("Beta", f"{beta:.2f}" if beta is not None else "N/A")

            with st.expander("Raw live data JSON"):
                st.json(payload)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transcripts", f"{filtered['doc_id'].nunique():,}" if not filtered.empty else "0")
    c2.metric("Companies", f"{filtered['symbol'].nunique()}" if not filtered.empty else "0")
    c3.metric("Avg Sentiment", f"{filtered['sentiment'].mean():+.3f}" if not filtered.empty else "N/A")
    c4.metric("Topics", f"{filtered['topic_id'].nunique()}" if not filtered.empty else "0")
    st.divider()

    tab_chat, tab_sent, tab_exp = st.tabs(["💬 Ask", "📈 Sentiment", "🔍 Explorer"])

    with tab_chat:
        st.subheader("Ask a Question")
        examples = [
            "What is NVDA saying about AI datacenter capex?",
            "How is MSFT positioning Azure AI revenue?",
            "What supply chain challenges did AAPL mention?",
        ]
        st.write("**Try:** " + " | ".join(f"`{e}`" for e in examples))

        q = st.text_input("Question", placeholder="e.g., What is NVDA saying about AI datacenter capex?")
        top_k = st.slider("Top-K results", 3, 10, 5)

        include_live_data = st.toggle(
            "Include live data",
            value=False,
            help="Optional: show current company fundamentals as supplemental context alongside the transcript answer.",
        )

        if st.button("🔍 Retrieve & Generate", type="primary") and q:
            with st.spinner("Retrieving..."):
                r = get_retriever()
                t_id = None
                if sel_topic != "All":
                    t_id = int(
                        topic_list.loc[
                            topic_list["topic_name"].astype(str) == str(sel_topic),
                            "topic_id",
                        ].iloc[0]
                    )

                inferred_symbol = None
                if len(sel_symbol) == 1 and sel_symbol[0]:
                    inferred_symbol = sel_symbol[0]
                elif symbols:
                    inferred_symbol = _extract_symbol(q, allowed_symbols=symbols)

                ctx = [] if r is None else r.retrieve(
                    q,
                    top_k=top_k,
                    symbol=inferred_symbol,
                    year=int(sel_year) if sel_year != "All" else None,
                    topic_id=t_id,
                    sentiment_min=sent_range[0],
                    sentiment_max=sent_range[1],
                )

            if not ctx:
                if include_live_data:
                    with st.spinner("Fetching live data..."):
                        ans, tool_log = generate_answer_with_tools(
                            q,
                            ctx,
                            force_tool=True,
                            allowed_symbols=symbols if symbols else None,
                        )
                    render_live_data_cards(tool_log)
                    st.success(ans)
                    st.info("No transcript context matched this query, so the result is based on live market data only.")
                else:
                    st.warning("No transcript matches found for this query. Try a broader question or relax filters.")
            else:
                st.markdown("**Retrieved Sources:**")
                for i, c in enumerate(ctx, 1):
                    text = str(c.get("text", ""))
                    record = {
                        "symbol": c.get("symbol", "N/A"),
                        "quarter": c.get("quarter", "N/A"),
                        "year": c.get("year", "N/A"),
                        "sentiment": c.get("sentiment", 0.0),
                        "topic": c.get("topic", "N/A"),
                        "score": c.get("score", 0.0),
                    }
                    with st.expander(
                        f"[{record['symbol']} Q{record['quarter']} {record['year']}] Sentiment: {record['sentiment']:+.2f} | Topic: {record['topic']} | Score: {record['score']}",
                        expanded=(i == 0),
                    ):
                        st.write(text[:800] + ("..." if len(text) > 800 else ""))

                st.markdown("**Answer:**")
                with st.spinner("Generating..."):
                    ans = generate_answer(q, ctx)
                st.success(ans)

                if include_live_data:
                    with st.spinner("Fetching live data..."):
                        _, tool_log = generate_answer_with_tools(
                            q,
                            ctx,
                            force_tool=True,
                            allowed_symbols=symbols if symbols else None,
                        )
                    render_live_data_cards(tool_log)

    with tab_sent:
        st.subheader("FinBERT Sentiment Analysis")
        if filtered.empty:
            st.info("No chunk data matches the current filters, so no sentiment charts are available yet.")
        elif px is not None:
            plot_df = filtered[["sentiment"]].dropna().copy()
            if plot_df.empty:
                st.info("No valid sentiment values are available for the selected filters.")
            else:
                plot_df["sentiment_label"] = plot_df["sentiment"].apply(
                    lambda s: "positive" if s > 0.15 else "negative" if s < -0.15 else "neutral"
                )
                ca, cb = st.columns(2)
                with ca:
                    fig = px.histogram(
                        plot_df,
                        x="sentiment",
                        color="sentiment_label",
                        nbins=40,
                        color_discrete_map={
                            "positive": "#22c55e",
                            "neutral": "#6b7280",
                            "negative": "#ef4444",
                        },
                        category_orders={"sentiment_label": ["negative", "neutral", "positive"]},
                        title="Score Distribution",
                        opacity=0.8,
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        height=350,
                        margin={"l": 20, "r": 20, "t": 40, "b": 20},
                        legend={"title_text": "Sentiment", "orientation": "h", "y": 1.1, "x": 0.0},
                    )
                    fig.update_xaxes(title_text="sentiment")
                    fig.update_yaxes(title_text="count")
                    st.plotly_chart(fig, use_container_width=True)
                with cb:
                    top_cos = filtered["symbol"].value_counts().head(10).index.tolist()
                    company_df = filtered[filtered["symbol"].isin(top_cos)][["symbol", "sentiment"]].dropna()
                    if company_df.empty:
                        st.info("No company sentiment values are available for the current selection.")
                    else:
                        fig = px.box(
                            company_df,
                            x="symbol",
                            y="sentiment",
                            title="By Company",
                            color_discrete_sequence=["#6366f1"],
                        )
                        fig.update_layout(template="plotly_dark", height=350, margin={"l": 20, "r": 20, "t": 40, "b": 20})
                        st.plotly_chart(fig, use_container_width=True)

                yearly = filtered.groupby("year", dropna=False)["sentiment"].mean().reset_index()
                if not yearly.empty:
                    fig = px.line(
                        yearly,
                        x="year",
                        y="sentiment",
                        markers=True,
                        title="Average Sentiment Trend",
                        color="year",
                        color_discrete_sequence=["#8b5cf6"],
                    )
                    fig.update_layout(showlegend=False, template="plotly_dark", height=300, margin={"l": 20, "r": 20, "t": 40, "b": 20})
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Plotly is unavailable in this environment.")

    with tab_exp:
        st.subheader("Data Explorer")
        if filtered.empty:
            st.info("No data matches the current filters.")
        else:
            view = st.radio("View", ["Table", "JSON"], horizontal=True)
            n = st.slider("Rows to show", 5, 50, 20)
            explorer_df = filtered[["symbol", "year", "quarter", "sentiment", "topic_name", "text"]].copy()
            explorer_df["sentiment"] = pd.to_numeric(explorer_df["sentiment"], errors="coerce")
            explorer_df["text"] = explorer_df["text"].fillna("").astype(str)
            if view == "Table":
                st.dataframe(
                    explorer_df.head(n),
                    use_container_width=True,
                    height=500,
                )
            else:
                st.json(explorer_df.head(n).to_dict("records"))
            st.download_button(
                "⬇️ Download CSV (first 1000 rows)",
                explorer_df.head(1000).to_csv(index=False).encode("utf-8"),
                "filtered_chunks.csv",
                "text/csv",
            )

    st.divider()
    st.caption(f"Powered by: Streamlit + FinBERT + FAISS + BM25 + Cross-Encoder + Groq ({GROQ_MODEL})")