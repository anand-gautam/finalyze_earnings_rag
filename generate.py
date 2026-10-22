# generate.py
import os
import json
import re
import sys
from pathlib import Path

# ROOT = Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from config import GROQ_MODEL
from tools import TOOL_SCHEMA, dispatch

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

try:
    from litellm import completion
    HAS_LITELLM = True
except ImportError:
    completion = None
    litellm = None
    HAS_LITELLM = False


load_dotenv()

DEFAULT_GROQ_MODEL = GROQ_MODEL


def normalize_groq_model_name(model_name):
    if not model_name:
        return DEFAULT_GROQ_MODEL
    normalized = str(model_name).strip()
    # ChatGroq wants 'openai/gpt-oss-120b', NOT 'groq/openai/gpt-oss-120b'
    if normalized.startswith("groq/"):
        normalized = normalized[len("groq/"):]
    return normalized or DEFAULT_GROQ_MODEL

    # if "/" in normalized:
    #     normalized = normalized.split("/", 1)[-1].strip()
    # if "gpt-oss" in normalized.lower():
    #     print("[WARN] Groq does not support gpt-oss models; using llama-3.3-70b-versatile instead.")
    #     return DEFAULT_GROQ_MODEL
    # return normalized or DEFAULT_GROQ_MODEL

def _build_context_str(context):
    return "\n\n".join(
        f"[{ctx.get('symbol', 'N/A')}"
        f"Q{ctx.get('quarter', 'N/A')}"
        f"{ctx.get('year', 'N/A')} | "
        f"Topic: {ctx.get('topic', 'N/A')} | "
        f"Sentiment: {ctx.get('sentiment', 0.0)}\n"
        f"{ctx.get('text', '')}"
        for ctx in context
    )

SYSTEM_RAG_TOOLS = (
    "You are an equity research assistant answering questions about S&P 500 companies.\n"
    "You are given retrieved earnings-call excerpts as CONTEXT, and you may call a tool to fetch CURRENT market "
    "fundamentals.\n\n"
    "Rules:\n"
    "1. Historical statements, guidance, and management commentary must come ONLY from the context.\n"
    "2. If the question concerns present-day price, valuation, market cap, dividend yield or financial health, "
    "call fetch_fundamentals before answering. Do NOT call it for questions phrased in past tense about a "
    "specific quarter or year.\n"
    "3. Never invent numbers. If neither context nor the tool provides a figure, say so.\n"
    "4. Cite transcript facts inline as [SYMBOL Q# YYYY] and tool results as [live market data].\n"
    "5. When both are used, clearly separate historical information from current information.\n"
    "6. Keep the final answer under 250 words."
)

def generate_answer(query, context):
    if not context:
        return "No relevant earnings call context was retrieved for this question."
    context_str = _build_context_str(context)
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key or ChatGroq is None or ChatPromptTemplate is None or StrOutputParser is None:
        return (
            "[Extracted Earnings Call Context (set GROQ_API_KEY in .env for LLM synthesis)]\n\n"
            + context_str[:2500]
            + ("\n\n..." if len(context_str) > 2500 else "")
        )

    try:
        groq_model = normalize_groq_model_name(os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL))
        llm = ChatGroq(model=groq_model, temperature=0.1, groq_api_key=groq_api_key)
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert financial analyst. Anwer the user's question accurately using ONLY the provided"
                "earnings call context. Cite sources using [SYMBOL Q# YYYY]. If the information is insufficient, "
                "state that clearly."
                "Keep answer under 250 words.",
            ),
            ("human", "Context Excerpts:\n{context}\n\nQuestion: {question}"),
        ])
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context_str, "question": query})
    except Exception as e:
        return f"[Error invoking LLM ({e}). Displaying retrieved context]: \n\n{context_str[:2500]}"


if tool is not None:
    @tool
    def fetch_fundamentals(symbol: str, include_income_statement: bool = False) -> str:
        """Fetch CURRENT market data, valuation ratios, and profitability metrics for a
        public company. Use this when the user asks about present-day stock price, market
        cap, P/E ratio, dividend yield, analyst targets, or financial health. Do NOT use
        this for historical earnings-call content or past quarterly results."""
        return dispatch(
            "get_company_fundamentals",
            {"symbol": symbol, "include_income_statement": include_income_statement},
        )
else:
    fetch_fundamentals = None

# Backward-compatible alias for older call sites.
get_company_fundamentals = fetch_fundamentals

# def generate_answer_with_tools(query, context, max_tol_rounds=3, force_tool=False):
#     """
#     Tool-aware generation. Returns (answer_text, tool_log).
#     tool_log_entries: {"tool": name, "args": str, "result": josn_str}
#     """
#     context_str = _build_context_str(context) if context else "(no transcript context)"
#     groq_api_key = os.environ.get("GROQ_API_KEY")
#
#     if not HAS_LITELLM or not groq_api_key:
#         fallback = generate_answer(query, context)
#         return fallback, []
#     model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
#     if not model.startswith("groq/"):
#         model = f"groq/{model}"
#
#         messages = [
#             {"role": "system", "context": SYSTEM_RAG_TOOLS},
#             {"role": "user", "context": f"CONTEXT:\n{context_str}\n\nQUESTION: {query}"},
#         ]
#
#         tool_log = []
#         last_text = ""
#
#         for rnd in range(max_tol_rounds):
#             choice = "requried" if (force_tool and rnd == 0) else "auto"
#             try:
#                 resp = completion(
#                     model=model,
#                     messages=messages,
#                     tools=[TOOL_SCHEMA],
#                     tool_choice=choice,
#                     temperature=0.0,
#                     max_tokens=900,
#                     api_key=groq_api_key,
#                 )
#             except Exception as e:
#                 # Provider rejected tool params -> degrade to plain generation
#                 print(f"[tools] completion failed ({type(e).__name__}: {e}); falling back")
#                 return generate_answer(query, context), tool_log
#             msg = resp.choices[0].message
#             last_text = (msg.content or "").strip()
#             calls = getattr(msg, "tool_calls", None)
#             if not calls:
#                 return last_text or "(empty response", tool_log
#
#             messages.append({
#                 "role": "assistant",
#                 "content": msg.content or "",
#                 "tool_calls": [
#                     {
#                         "id": tc.id,
#                         "type": "function",
#                         "function": {
#                             "name": tc.function.name,
#                             "arguments": tc.function.arguments,
#                         },
#                     }
#                     for tc in calls
#                 ],
#
#             }
#             )
#
#             for tc in calls:
#                 payload = dispatch(tc.function.name, tc.function.arguments)
#                 tool_log.append(
#                     {
#                         "tool": tc.function.name,
#                         "args": tc.function.arguments,
#                         "result": payload,
#                     }
#                 )
#                 messages.append({
#                     "role": "tool",
#                     "tool_call_id": tc.id,
#                     "content": payload,
#                 })
#
#         # if rounds exhausted - force a text-only summary
#         messages.append({
#             "role": "user",
#             "content": "Do nnot call any more tools. Answer the question now using the context and tool results"
#                        "already provided.",
#         })
#         try:
#             final = completion(
#                 model=model,
#                 messages=messages,
#                 temperature=0.0,
#                 max_tokens=700,
#                 tool_choice="none",
#                 api_key=groq_api_key,
#             )
#             return (final.choices[0].message or last_text).strip(), tool_log
#         except Exception as e:
#             return last_text or "(could not synthesize final answer)", tool_log

def _extract_symbol(query: str, allowed_symbols=None):
    text = str(query or "")
    upper = text.upper()
    matches = re.findall(r"\b[A-Z]{1,6}\.?[A-Z0-9\-]{0,5}\b", upper)

    if allowed_symbols:
        allowed = {str(s).strip().upper() for s in allowed_symbols if str(s).strip()}
        for m in matches:
            if m in allowed:
                return m
        for m in sorted(allowed):
            if re.search(rf"\b{re.escape(m)}\b", upper):
                return m
        return None

    for m in matches:
        if m not in {"AI", "Q", "US", "THE", "ETF", "EPS", "CEO", "CFO"}:
            return m

    for m in ["MSFT", "AAPL", "NVDA", "DELL", "AMZN", "GOOG", "GOOGL", "META", "BRK-B", "TSLA", "NFLX", "AMD"]:
        if re.search(rf"\b{re.escape(m)}\b", upper):
            return m
    return None


def generate_answer_with_tools(query, context, max_tool_rounds=3, force_tool=False, allowed_symbols=None):
    """Generate answer using live market data only when explicitly requested."""
    direct_live = False
    if force_tool:
        direct_live = True

    if direct_live:
        symbol = _extract_symbol(query, allowed_symbols=allowed_symbols)
        if not symbol:
            if allowed_symbols:
                return "The company is not listed in the scope of this app now.", []
            return "Live market data request requires a company ticker to be identified.", []

        if allowed_symbols:
            allowed = {str(s).strip().upper() for s in allowed_symbols if str(s).strip()}
            if symbol.upper() not in allowed:
                return "The company is not listed in the scope of this app now.", []

        payload = dispatch("get_company_fundamentals", {"symbol": symbol})
        tool_log = [{
            "tool": "get_company_fundamentals",
            "args": str({"symbol": symbol}),
            "result": payload,
        }]

        try:
            data = json.loads(payload)
        except Exception:
            data = {"raw": payload}

        if isinstance(data, dict) and data.get("error"):
            return f"Live market data request failed for {symbol}: {data.get('error')}", tool_log

        price = data.get("current_price")
        market_cap = data.get("market_cap")
        pe = data.get("trailing_pe")
        div_yield = data.get("dividend_yield_pct")

        parts = [
            f"price ${price:,.2f}" if price is not None else "price N/A",
            f"market cap ${market_cap:,.0f}" if market_cap is not None else "market cap N/A",
            f"trailing P/E {pe:.2f}" if pe is not None else "trailing P/E N/A",
            f"dividend yield {div_yield:.2f}%" if div_yield is not None else "dividend yield N/A",
        ]
        answer = f"{symbol} current market snapshot: " + "; ".join(parts) + "."
        return answer, tool_log

    context_str = _build_context_str(context) if context else "(no transcript context)"
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not ChatGroq or not fetch_fundamentals:
        return generate_answer(query, context), []

    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate

        groq_model = normalize_groq_model_name(os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL))
        llm = ChatGroq(model=groq_model, temperature=0.1, groq_api_key=groq_api_key)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_RAG_TOOLS),
            ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(llm, [fetch_fundamentals], prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=[fetch_fundamentals],
            verbose=False,
            max_iterations=max_tool_rounds,
        )

        result = executor.invoke({"context": context_str, "question": query})
        answer = result.get("output", "").strip()

        tool_log = []
        for step in result.get("intermediate_steps", []):
            action, observation = step
            if getattr(action, "tool", "") == "fetch_fundamentals":
                tool_log.append({
                    "tool": "get_company_fundamentals",
                    "args": str(action.tool_input),
                    "result": str(observation)[:2000],
                })

        return answer, tool_log

    except ImportError:
        print("[tools] langchain.agents unavailable; using transcript-only generation")
        return generate_answer(query, context), []
    except Exception as e:
        print(f"[tools] agent failed ({type(e).__name__}: {e}); falling back")
        return generate_answer(query, context), []

# LIVE_DATA_HINTS = (
#     "stock price", "share price", "market cap", "valuation", "p/e", "pe ratio",
#     "dividend yield", "today", "current", "now", "52-week", "52 week",
#     "analyst target", "financial health", "debt level", "overvalued", "undervalued",
#     "revenue", "ai revenue", "azure", "market share", "growth outlook", "financial health"
# )

LIVE_DATA_HINTS = (
    "live price", "current price", "stock price", "share price",
    "market cap", "trailing pe", "forward pe", "p/e", "pe ratio",
    "dividend yield", "valuation", "financial health", "analyst target",
    "beta", "today", "now", "52-week", "52 week"
)

NON_LIVE_CONTEXT_HINTS = (
    "positioning", "strategy", "strategic", "how is", "what is", "saying",
    "guidance", "outlook", "revenue", "azure ai", "ai revenue"
)


def needs_live_data(query: str) -> bool:
    q = str(query).lower().strip()
    if not q: return False
    contains_strategy_context = any(h in q for h in ("positioning", "strategy", "strategic", "outlook", "guidance"))
    if contains_strategy_context and not any(h in q for h in
                                             ("price", "market cap", "p/e", "pe ratio", "dividend yield", "valuation",
                                              "financial health", "beta")):
        return False
    return any(h in q for h in LIVE_DATA_HINTS)

REACT_PROMPT = """
    You are an equity research assistant. You may use ONE tool.
    
    To request live market data, reply with EXACTLY one line:
    ACTION: get_company_fundamentals(SYMBOL)
    
    Otherwise reply with:
    ANSWER: <your answer>
    
    Rules:
    - Use the tool only for present-day price, valuation, market cap, or financial health.
    - Answer historical/quarterly questions from the context alone.
    - Cite transcript facts as [SYMBOL Q# YYYY] and tool data as [live market data].
    
    CONTEXT:
    {ctx}
    
    QUESTION: {q}
    """

def generate_answer_react(query, context, max_steps=2):
    """Fallback when the provider doesn't support native function calling."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not HAS_LITELLM or not groq_api_key:
        return generate_answer(query, context), []
    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    if not model.startswith("groq/"):
        model = f"groq/{model}"

    ctx_block = _build_context_str(context[:6000]) if context else "(no transcript context)"
    prompt = REACT_PROMPT.format(ctx=ctx_block, query=query)
    tool_log = []
    out = ""

    for _ in range(max_steps):
        try:
            out = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=700,
                api_key=groq_api_key,
            ).choices[0].message_content or ""
        except Exception as e:
            return f"ReAct error: {e}", tool_log

        m = re.search(r"ACTION:\s*get_company_fundamentals\((\s*[A-Za-z.\-]{1,8)\s*\)", out)
        if not m:
            return re.sub(r"^.*ANSWER:\s*", "", out, flags=re.DOTALL).strip(), tool_log

        payload = dispatch("get_company_fundamentals", {"symbol": m.group(1)})
        tool_log.append({"tool": "get_company_fundamentals", "args": m.group(1), "result": payload})
        prompt += f"\n\nOBSERVATION: {payload}\nNow give your final ANSWER."
    return re.sub(r"^*?ANSWER:\s*", "", out, flags=re.DOTALL).strip(), tool_log

# def generate_answer(query, context):
#     if not context:
#         return "No relevant earnings call context was retrieved for this question."
#
#     context_str = "\n\n".join(
#         f"[{c.get('symbol', 'N/A')} Q{c.get('quarter', 'N/A')} {c.get('year', 'N/A')} | Topic: {c.get('topic', 'N/A')} | Sentiment: {c.get('sentiment', 0.0)}]\n{c.get('text', '')}"
#         for c in context
#     )
#
#     groq_api_key = os.environ.get("GROQ_API_KEY")
#
#     if not groq_api_key or ChatGroq is None or ChatPromptTemplate is None or StrOutputParser is None:
#         return (
#             "[Extracted Earnings Call Context (Set GROQ_API_KEY in .env for LLM synthesis)]\n\n"
#             + context_str[:2500]
#             + ("\n\n..." if len(context_str) > 2500 else "")
#         )
#
#     try:
#         groq_model = normalize_groq_model_name(os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL))
#         llm = ChatGroq(model=groq_model, temperature=0.1, groq_api_key=groq_api_key)
#
#         prompt = ChatPromptTemplate.from_messages([
#             (
#                 "system",
#                 "You are an expert financial analyst. Answer the user's question accurately using ONLY the provided earnings call context. "
#                 "Cite sources using [SYMBOL Q# YYYY]. If the information is insufficient, state that clearly. Keep answer under 250 words.",
#             ),
#             ("human", "Context Excerpts:\n{context}\n\nQuestion: {question}"),
#         ])
#
#         chain = prompt | llm | StrOutputParser()
#         return chain.invoke({"context": context_str, "question": query})
#     except Exception as e:
#         return (
#             f"[Error invoking LLM ({e}). Displaying retrieved context]:\n\n"
#             + context_str[:2500]
#         )