import os
import numpy as np
import pandas as pd
import faiss
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer

# Docker: ensure Ollama client connects to local server inside container
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FixIt — Repair Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH   = "faiss_index.bin"
CHUNKS_PATH  = "rag_chunks.parquet"
OLLAMA_MODEL = "mistral"
TOP_K        = 5
MAX_CONTEXT  = 4

DEVICES = [
    {"label": "Phone",             "source": "Phone",             "guides": 3682},
    {"label": "PC",                "source": "PC",                "guides": 3592},
    {"label": "Tablet",            "source": "Tablet",            "guides": 2272},
    {"label": "Mac",               "source": "Mac",               "guides": 2224},
    {"label": "Camera",            "source": "Camera",            "guides": 1717},
    {"label": "Electronics",       "source": "Electronics",       "guides": 1546},
    {"label": "Household",         "source": "Household",         "guides":  940},
    {"label": "Game Console",      "source": "Game Console",      "guides":  705},
    {"label": "Appliance",         "source": "Appliance",         "guides":  609},
    {"label": "Computer Hardware", "source": "Computer Hardware", "guides":  515},
]

SUGGESTIONS = {
    "Phone":             ["How do I replace the screen?", "How do I fix the battery?", "How do I replace the charging port?"],
    "PC":                ["How do I remove the hard drive?", "How do I clean the CPU fan?", "How do I upgrade the RAM?"],
    "Tablet":            ["How do I replace the battery?", "How do I fix a cracked screen?", "How do I replace the charging port?"],
    "Mac":               ["How do I replace the battery?", "How do I remove the keyboard?", "How do I open the case?"],
    "Camera":            ["How do I clean the lens?", "How do I replace the shutter?", "How do I fix the battery compartment?"],
    "Electronics":       ["How do I fix a broken component?", "How do I replace the power supply?", "How do I repair the circuit board?"],
    "Household":         ["How do I fix a broken hinge?", "How do I replace a broken part?", "How do I disassemble this device?"],
    "Game Console":      ["How do I open the console?", "How do I replace the disk drive?", "How do I fix the controller?"],
    "Appliance":         ["How do I replace the motor?", "How do I fix the power switch?", "How do I clean the filter?"],
    "Computer Hardware": ["How do I install a graphics card?", "How do I replace a cooling fan?", "How do I clean the motherboard?"],
}

SYSTEM_PROMPT = """You are a professional repair assistant. Answer device repair questions using ONLY the provided context.

FORMAT YOUR ANSWER EXACTLY LIKE THIS — no exceptions:

Tools you will need:
- [tool 1]
- [tool 2]
(Write "None specified" if no tools appear in the context)

Steps:
1. [clear instruction in plain English]
2. [clear instruction in plain English]
3. [clear instruction in plain English]

Sources:
- [Guide title] — Step [number]
- [Guide title] — Step [number]

RULES:
- Write each step as a plain English instruction. No guide names or step numbers inside the steps themselves.
- Keep instructions short and actionable. One action per step.
- Only include steps that appear in the context. Do not invent steps.
- List all guides you used under Sources at the end.
- If the context has no relevant information, write only: "No relevant repair steps found. Please try rephrasing your question."
"""

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Lato:wght@300;400;700&family=Source+Code+Pro:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #F5EBE0;
    color: #704241;
}

[data-testid="stAppViewContainer"] { background: #F5EBE0; }
[data-testid="stHeader"] { display: none; }
#MainMenu, footer, .stDeployButton { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }

.stTextInput > div > div > input {
    background: #fffaf6 !important;
    border: 1.5px solid #c4a99e !important;
    border-radius: 8px !important;
    color: #704241 !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 15px !important;
    padding: 14px 18px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    caret-color: #98A869;
}
.stTextInput > div > div > input:focus {
    border-color: #98A869 !important;
    box-shadow: 0 0 0 3px rgba(152,168,105,0.2) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: #c4a99e !important; }
.stTextInput label { display: none !important; }

.stButton > button {
    background: #704241 !important;
    color: #F5EBE0 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Lato', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.6px !important;
    padding: 11px 24px !important;
    transition: background 0.15s, transform 0.1s !important;
    cursor: pointer !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: #5a3333 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

.stSpinner > div { border-top-color: #704241 !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #F5EBE0; }
::-webkit-scrollbar-thumb { background: #c4a99e; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Pipeline ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    encoder = SentenceTransformer(EMBED_MODEL, device=device)
    index   = faiss.read_index(INDEX_PATH)
    rag_df  = pd.read_parquet(CHUNKS_PATH)
    return encoder, index, rag_df


def retrieve(query, encoder, index, rag_df, source_filter, k=TOP_K):
    if "source_file" not in rag_df.columns:
        return None  # signals missing column — handled by caller

    vec = encoder.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(vec)
    scores, indices = index.search(vec, min(k * 8, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = rag_df.iloc[idx]
        if row.get("source_file", "") == source_filter:
            results.append((float(score), int(idx)))
        if len(results) == k:
            break
    return results


def build_context(results, rag_df):
    parts = []
    for rank, (score, idx) in enumerate(results[:MAX_CONTEXT], 1):
        row   = rag_df.iloc[idx]
        tools = (
            ", ".join(row["tools_final"])
            if isinstance(row["tools_final"], list) and row["tools_final"]
            else "not specified"
        )
        parts.append(
            f"[Source {rank} | similarity {score:.4f}]\n"
            f"Guide      : {row['title']}\n"
            f"Subject    : {row['subject']}\n"
            f"Step       : {row['step_order']}\n"
            f"Tools      : {tools}\n"
            f"Instruction: {row['text_clean']}"
        )
    return "\n\n---\n\n".join(parts)


def stream_answer(query, context):
    stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Repair context:\n{context}\n\nQuestion: {query}"},
        ],
        options={"temperature": 0.0, "num_predict": 600, "repeat_penalty": 1.15},
        stream=True,
    )
    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token


# ── Session state ──────────────────────────────────────────────────────────────
if "selected_device" not in st.session_state:
    st.session_state.selected_device = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Load pipeline ──────────────────────────────────────────────────────────────
artifacts_exist = os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)
err = ""

if artifacts_exist:
    with st.spinner("Loading pipeline..."):
        try:
            encoder, index, rag_df = load_pipeline()
            pipeline_ok = True
        except Exception as e:
            pipeline_ok = False
            err = str(e)
else:
    pipeline_ok = False
    err = "Index files not found. Run Cells 1–8 in the notebook to build the index."

# ══════════════════════════════════════════════════════════════════════════════
#  VIEW A — DEVICE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.selected_device is None:

    # Hero
    st.markdown("""
    <div style="max-width:840px;margin:0 auto;padding:72px 40px 0">
        <div style="font-family:'Source Code Pro',monospace;font-size:10px;font-weight:500;
                    letter-spacing:4px;color:#98A869;text-transform:uppercase;margin-bottom:14px">
            MyFixit — Repair Intelligence
        </div>
        <h1 style="font-family:'Playfair Display',serif;font-size:clamp(34px,5vw,50px);
                   font-weight:700;color:#704241;line-height:1.15;margin:0 0 20px">
            What are you<br/>trying to fix?
        </h1>
        <p style="font-size:15px;color:#9b6a68;font-weight:300;line-height:1.8;
                  margin:0 0 56px;max-width:460px">
            Select a device category. Our assistant will search through 18,000+ real
            repair guides and give you step-by-step instructions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Device grid
    st.markdown('<div style="max-width:840px;margin:0 auto;padding:0 40px">', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    for i, device in enumerate(DEVICES):
        col = col_a if i % 2 == 0 else col_b
        with col:
            label = f"{device['label']}  —  {device['guides']:,} guides"
            if st.button(label, key=f"dev_{device['source']}", use_container_width=True):
                if pipeline_ok:
                    st.session_state.selected_device = device
                    st.session_state.chat_history    = []
                    st.rerun()
                else:
                    st.error(f"Pipeline not ready: {err}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Status bar
    st.markdown('<div style="max-width:840px;margin:0 auto;padding:0 40px">', unsafe_allow_html=True)

    if pipeline_ok:
        n_chunks = len(rag_df)
        n_guides = rag_df["guidid"].nunique()
        has_filter = "source_file" in rag_df.columns
        filter_note = "Device filtering active" if has_filter else "Rebuild index to enable device filtering"
        st.markdown(f"""
        <div style="margin-top:48px;padding:18px 24px;border-top:1px solid rgba(112,66,65,0.15);
                    display:flex;align-items:center;gap:40px">
            <div>
                <div style="font-family:'Source Code Pro',monospace;font-size:10px;
                            color:#98A869;letter-spacing:1px;text-transform:uppercase;
                            margin-bottom:4px">Index status</div>
                <div style="font-size:13px;color:#704241;font-weight:700">{n_chunks:,} steps indexed</div>
            </div>
            <div>
                <div style="font-family:'Source Code Pro',monospace;font-size:10px;
                            color:#9b6a68;letter-spacing:1px;text-transform:uppercase;
                            margin-bottom:4px">Coverage</div>
                <div style="font-size:13px;color:#704241">{n_guides:,} guides · 15 categories</div>
            </div>
            <div>
                <div style="font-family:'Source Code Pro',monospace;font-size:10px;
                            color:#9b6a68;letter-spacing:1px;text-transform:uppercase;
                            margin-bottom:4px">Filtering</div>
                <div style="font-size:13px;color:{'#98A869' if has_filter else '#c4a99e'}">{filter_note}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="margin-top:40px;padding:16px 20px;border:1.5px solid rgba(112,66,65,0.25);
                    border-radius:8px;background:rgba(112,66,65,0.05)">
            <div style="font-size:13px;font-weight:700;color:#704241;margin-bottom:4px">
                Pipeline not ready
            </div>
            <div style="font-size:13px;color:#9b6a68">{err}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  VIEW B — CHAT
# ══════════════════════════════════════════════════════════════════════════════
else:
    device = st.session_state.selected_device

    # Header
    st.markdown(f"""
    <div style="background:#704241;padding:22px 44px;
                display:flex;align-items:center;gap:20px">
        <div style="width:36px;height:36px;border-radius:6px;
                    background:rgba(245,235,224,0.15);
                    display:flex;align-items:center;justify-content:center">
            <div style="width:14px;height:14px;border:2px solid #F5EBE0;border-radius:2px"></div>
        </div>
        <div>
            <div style="font-family:'Playfair Display',serif;font-size:18px;
                        font-weight:600;color:#F5EBE0;letter-spacing:0.2px">
                {device['label']} Repair Assistant
            </div>
            <div style="font-family:'Source Code Pro',monospace;font-size:10px;
                        color:rgba(245,235,224,0.6);margin-top:3px;letter-spacing:1px">
                {device['guides']:,} GUIDES AVAILABLE
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Back button
    st.markdown('<div style="padding:20px 44px 0">', unsafe_allow_html=True)
    if st.button("Back to devices", key="back_btn"):
        st.session_state.selected_device = None
        st.session_state.chat_history    = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Divider
    st.markdown("""
    <div style="max-width:780px;margin:28px auto 0;padding:0 40px">
        <div style="height:1px;background:rgba(112,66,65,0.12)"></div>
    </div>
    """, unsafe_allow_html=True)

    # Chat history
    st.markdown('<div style="max-width:780px;margin:0 auto;padding:0 40px">', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown(f"""
        <div style="padding:48px 0 24px;text-align:center">
            <div style="font-family:'Playfair Display',serif;font-size:22px;
                        color:#704241;margin-bottom:10px">
                Ask anything about {device['label']} repairs
            </div>
            <div style="font-size:14px;color:#9b6a68;font-weight:300">
                Questions are answered using only verified repair guides from the MyFixit dataset.
            </div>
        </div>
        """, unsafe_allow_html=True)

    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-end;margin:24px 0 10px">
                <div style="background:#704241;color:#F5EBE0;
                            border-radius:16px 16px 4px 16px;
                            padding:14px 20px;max-width:72%;
                            font-size:14px;line-height:1.65;
                            font-family:'Lato',sans-serif;font-weight:400">
                    {turn['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-start;margin:10px 0 6px">
                <div style="background:#fffaf6;
                            border:1px solid rgba(112,66,65,0.12);
                            border-left:3px solid #98A869;
                            border-radius:4px 16px 16px 16px;
                            padding:22px 26px;max-width:88%;
                            font-size:14px;line-height:1.85;
                            color:#4a2e2e;font-family:'Lato',sans-serif;
                            white-space:pre-wrap">
{turn['content']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Source cards
            if turn.get("sources"):
                cards_html = ""
                for rank, src in enumerate(turn["sources"], 1):
                    pct        = int(src["score"] * 100)
                    bar_color  = "#98A869" if pct >= 60 else "#c4a99e"
                    title_safe = src["title"][:75] + ("..." if len(src["title"]) > 75 else "")
                    step_str   = f"Step {src['step']}"
                    subj_str   = f" · {src['subject']}" if src["subject"] else ""
                    cards_html += f"""
                    <div style="background:#f0e8e0;border-radius:6px;
                                padding:12px 16px;margin-bottom:8px">
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:8px">
                            <span style="font-family:'Source Code Pro',monospace;
                                         font-size:9px;letter-spacing:1.5px;
                                         color:#9b6a68;text-transform:uppercase">
                                Source {rank}
                            </span>
                            <span style="font-family:'Source Code Pro',monospace;
                                         font-size:11px;font-weight:500;
                                         color:{bar_color}">{pct}%</span>
                        </div>
                        <div style="width:100%;height:2px;
                                    background:rgba(112,66,65,0.12);
                                    border-radius:2px;margin-bottom:10px">
                            <div style="width:{pct}%;height:2px;
                                        background:{bar_color};border-radius:2px"></div>
                        </div>
                        <div style="font-size:12px;font-weight:700;
                                    color:#704241;margin-bottom:3px">
                            {title_safe}
                        </div>
                        <div style="font-family:'Source Code Pro',monospace;
                                    font-size:10px;color:#9b6a68">
                            {step_str}{subj_str}
                        </div>
                    </div>"""

                st.markdown(f"""
                <div style="margin:4px 0 24px 4px">
                    <div style="font-family:'Source Code Pro',monospace;font-size:9px;
                                letter-spacing:2px;color:#9b6a68;text-transform:uppercase;
                                margin-bottom:10px">Retrieved sources</div>
                    {cards_html}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Input area
    st.markdown('<div style="max-width:780px;margin:20px auto 0;padding:0 40px 48px">', unsafe_allow_html=True)

    # Suggestion chips
    chips = SUGGESTIONS.get(device["label"], [])
    if chips and not st.session_state.chat_history:
        st.markdown("""
        <div style="margin-bottom:12px">
            <span style="font-family:'Source Code Pro',monospace;font-size:9px;
                         letter-spacing:2px;color:#9b6a68;text-transform:uppercase">
                Suggested questions
            </span>
        </div>
        """, unsafe_allow_html=True)
        chip_cols = st.columns(len(chips))
        for col, chip in zip(chip_cols, chips):
            with col:
                if st.button(chip, key=f"chip_{chip}", use_container_width=True):
                    st.session_state["_prefill"] = chip
                    st.rerun()

    # Query input
    query_val = st.session_state.pop("_prefill", "")
    query_input = st.text_input(
        label="",
        value=query_val,
        placeholder=f"Ask a repair question about your {device['label']}...",
        key="query_input",
        label_visibility="collapsed",
    )

    btn_col, clear_col, _ = st.columns([1, 1, 4])
    with btn_col:
        ask_clicked = st.button("Ask", key="ask_btn", use_container_width=True)
    with clear_col:
        if st.button("Clear", key="clear_btn", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    active_query = query_input.strip() if ask_clicked and query_input.strip() else None

    if active_query and pipeline_ok:
        st.session_state.chat_history.append({"role": "user", "content": active_query})

        with st.spinner(f"Searching {device['guides']:,} {device['label']} guides..."):
            results = retrieve(active_query, encoder, index, rag_df, device["source"], k=TOP_K)

        # Handle missing source_file column
        if results is None:
            full_answer = (
                "Device filtering is unavailable because your index was built before "
                "this feature was added. Please delete faiss_index.bin and "
                "rag_chunks.parquet, then re-run Cells 1-8 in the notebook to rebuild."
            )
            sources = []

        elif not results:
            full_answer = (
                "No relevant repair steps found for this query. "
                "Please try rephrasing your question."
            )
            sources = []

        else:
            context = build_context(results, rag_df)
            placeholder = st.empty()
            full_answer = ""

            for token in stream_answer(active_query, context):
                full_answer += token
                placeholder.markdown(f"""
                <div style="background:#fffaf6;
                            border:1px solid rgba(112,66,65,0.12);
                            border-left:3px solid #98A869;
                            border-radius:4px 16px 16px 16px;
                            padding:22px 26px;font-size:14px;
                            line-height:1.85;color:#4a2e2e;
                            font-family:'Lato',sans-serif;
                            white-space:pre-wrap;margin:10px 0">
{full_answer}<span style="opacity:0.4">|</span></div>
                """, unsafe_allow_html=True)

            # Final render without cursor — no .empty() to avoid flicker
            placeholder.markdown(f"""
            <div style="background:#fffaf6;
                        border:1px solid rgba(112,66,65,0.12);
                        border-left:3px solid #98A869;
                        border-radius:4px 16px 16px 16px;
                        padding:22px 26px;font-size:14px;
                        line-height:1.85;color:#4a2e2e;
                        font-family:'Lato',sans-serif;
                        white-space:pre-wrap;margin:10px 0">
{full_answer}</div>
            """, unsafe_allow_html=True)

            sources = []
            for score, idx in results[:MAX_CONTEXT]:
                row = rag_df.iloc[idx]
                sources.append({
                    "score":   float(score),
                    "title":   str(row["title"]),
                    "step":    int(row["step_order"]) if pd.notna(row.get("step_order")) else "—",
                    "subject": str(row.get("subject", "") or ""),
                })

        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": full_answer,
            "sources": sources,
        })
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
