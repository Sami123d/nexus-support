# app.py

import streamlit as st
import time
import os
import re
from datetime import datetime
# Streamlit Cloud: copy keys from st.secrets into the environment *before*
# the agent module reads them at import time.
for _k in ("GEMINI_API_KEY", "DEEPSEEK_API_KEY", "FALLBACK_API_KEY", "PRIMARY_MODEL"):
    try:
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k]).strip()
    except Exception:
        pass  # no secrets.toml locally

from customer_support_agent import CustomerSupportAgent, llm_provider
from langchain_core.messages import HumanMessage, AIMessage

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Nexus Support · AI Customer Support Agent",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DESIGN SYSTEM (clean light SaaS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

    :root {
        --brand: #4F46E5;
        --brand-2: #7C3AED;
        --brand-soft: #EEF2FF;
        --bg: #F7F8FC;
        --surface: #FFFFFF;
        --border: #E5E7EB;
        --text: #0F172A;
        --muted: #64748B;
        --ok: #059669;
        --ok-soft: #ECFDF5;
        --shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -10px rgba(15,23,42,.12);
    }

    .stApp {
        background:
            radial-gradient(1100px 500px at 85% -10%, rgba(124,58,237,.08), transparent 60%),
            radial-gradient(900px 500px at -10% 0%, rgba(79,70,229,.08), transparent 60%),
            var(--bg);
        color: var(--text);
        font-family: 'Inter', system-ui, sans-serif;
    }
    h1, h2, h3, h4 { font-family: 'Plus Jakarta Sans', 'Inter', sans-serif; color: var(--text); letter-spacing: -0.02em; }
    .block-container { padding-top: 2.2rem; max-width: 1100px; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
    [data-testid="stSidebarNav"] { display: none; }
    .brand { display:flex; align-items:center; gap:.6rem; padding:.25rem 0 1.25rem; border-bottom:1px solid var(--border); margin-bottom:1.25rem; }
    .brand-mark { width:36px; height:36px; border-radius:10px; display:grid; place-items:center; color:#fff; font-weight:800;
                  background: linear-gradient(135deg, var(--brand), var(--brand-2)); box-shadow: 0 6px 16px -6px rgba(79,70,229,.6); }
    .brand-name { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.05rem; color:var(--text); line-height:1.1; }
    .brand-sub { font-size:.75rem; color:var(--muted); }
    .side-label { font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); margin:1.25rem 0 .6rem; }

    .card { background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:.9rem 1rem; margin-bottom:.6rem; box-shadow: var(--shadow); }
    .kv { display:flex; justify-content:space-between; align-items:center; font-size:.85rem; color:var(--muted); }
    .kv b { color:var(--text); font-weight:600; }
    .pill { display:inline-flex; align-items:center; gap:.35rem; padding:.2rem .6rem; border-radius:999px; font-size:.72rem; font-weight:600; }
    .pill-ok { background:var(--ok-soft); color:var(--ok); }
    .pill-brand { background:var(--brand-soft); color:var(--brand); }
    .dot { width:7px; height:7px; border-radius:50%; background:currentColor; display:inline-block; }

    /* Hero */
    .hero-badge { display:inline-flex; gap:.4rem; align-items:center; padding:.3rem .75rem; border-radius:999px; background:var(--surface);
                  border:1px solid var(--border); font-size:.78rem; font-weight:600; color:var(--brand); box-shadow:var(--shadow); }
    .hero-title { font-size:2.6rem; font-weight:800; margin:.8rem 0 .3rem; line-height:1.1; }
    .hero-title span { background:linear-gradient(135deg,var(--brand),var(--brand-2)); -webkit-background-clip:text; background-clip:text; color:transparent; }
    .hero-sub { color:var(--muted); font-size:1.05rem; margin-bottom:1.2rem; max-width:720px; }

    .agents { display:grid; grid-template-columns:repeat(5,1fr); gap:.6rem; margin:.4rem 0 1.4rem; }
    .agent { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:.7rem .8rem; box-shadow:var(--shadow); }
    .agent .ic { font-size:1.1rem; }
    .agent .nm { font-weight:600; font-size:.85rem; color:var(--text); margin-top:.2rem; }
    .agent .ds { font-size:.72rem; color:var(--muted); }
    .agent.on { border-color:var(--brand); background:var(--brand-soft); box-shadow:0 0 0 3px rgba(79,70,229,.12); }
    @media (max-width: 900px) { .agents { grid-template-columns:repeat(2,1fr); } .hero-title { font-size:2rem; } }

    /* Chat */
    [data-testid="stChatMessage"] { background:transparent; padding:.35rem 0; gap:.75rem; }
    [data-testid="stChatMessage"] { align-items: flex-start; }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { margin: 0; }
    [data-testid="stChatMessageContent"] { height: auto !important; min-height: 0 !important; }
    [data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessageContent"] .stMarkdown { margin-bottom: 0 !important; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
        background:var(--surface); border:1px solid var(--border); border-radius:4px 16px 16px 16px; padding:.75rem 1rem; box-shadow:var(--shadow); color:var(--text);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row-reverse; justify-content: flex-start; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background:linear-gradient(135deg,var(--brand),var(--brand-2)); color:#fff; border-radius:16px 4px 16px 16px; padding:.75rem 1rem;
        box-shadow:0 8px 20px -10px rgba(79,70,229,.6); max-width: 75%; flex: 0 1 auto; margin-left: auto !important; margin-right: 0 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] * { color:#fff !important; }
    [data-testid="stChatMessageAvatarAssistant"] { background:linear-gradient(135deg,var(--brand),var(--brand-2)) !important; color:#fff; }
    [data-testid="stChatMessageAvatarUser"] { background:#0F172A !important; color:#fff; }
    [data-testid="stBottom"] > div { background: transparent; }
    [data-testid="stChatInput"] { border:1px solid var(--border) !important; border-radius:14px !important; background:var(--surface) !important; box-shadow:var(--shadow); }

    /* Buttons */
    .stButton > button { border-radius:10px; border:1px solid var(--border); background:var(--surface); color:var(--text); font-weight:500; box-shadow: var(--shadow); }
    .stButton > button:hover { border-color:var(--brand); color:var(--brand); }

    .paused { background:#FEF2F2; border:1px solid #FECACA; color:#991B1B; border-radius:14px; padding:1rem 1.25rem; margin-bottom:1rem; }
    .paused b { color:#B91C1C; }

    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    .stDeployButton { display:none; }
</style>
""", unsafe_allow_html=True)

AGENTS = [
    ("supervisor", "🧭", "Supervisor", "Routes each message"),
    ("order_specialist", "📦", "Orders", "Tracking & status"),
    ("tech_specialist", "🛠️", "Tech", "Login & bugs"),
    ("billing_specialist", "💳", "Billing", "Payments & refunds"),
    ("escalate", "🧑‍💼", "Human", "Escalation & tickets"),
]
DEMO_PROMPTS = [
    "My email is alice@example.com. Where is my order?",
    "I can't log in, I forgot my password",
    "I want a refund for a double charge",
]

# --- INIT AGENT ---
@st.cache_resource
def load_agent():
    return CustomerSupportAgent()

agent = load_agent()

# --- SESSION STATE ---
if "state" not in st.session_state:
    st.session_state.state = None
if "history" not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div class="brand">
            <div class="brand-mark">N</div>
            <div><div class="brand-name">Nexus Support</div><div class="brand-sub">AI customer support agent</div></div>
        </div>
    """, unsafe_allow_html=True)

    s = st.session_state.state or {}
    agent_name = (s.get('active_agent') or 'supervisor').replace('_', ' ').title()
    customer = s.get('customer_name') or 'Not identified yet'
    tier = (s.get('customer_tier') or 'standard').title()
    engine = getattr(llm_provider.primary, "model_name", "not configured")
    db_kind = "Postgres" if os.getenv("DATABASE_URL", "").startswith("postgres") else "SQLite"

    st.markdown('<div class="side-label">Live session</div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="card">
            <div class="kv"><span>Active agent</span><span class="pill pill-brand"><span class="dot"></span>{agent_name}</span></div>
        </div>
        <div class="card">
            <div class="kv" style="margin-bottom:.35rem"><span>Customer</span><b>{customer}</b></div>
            <div class="kv"><span>Tier</span><b>{tier}</b></div>
        </div>
        <div class="card">
            <div class="kv"><span>Tokens used</span><b>{int(s.get('total_tokens', 0))}</b></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-label">Under the hood</div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="card">
            <div class="kv" style="margin-bottom:.35rem"><span>Model</span><b>{engine}</b></div>
            <div class="kv" style="margin-bottom:.35rem"><span>Database</span><b>{db_kind}</b></div>
            <div class="kv"><span>PII masking</span><span class="pill pill-ok"><span class="dot"></span>On</span></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-label">Demo data</div>', unsafe_allow_html=True)
    st.caption("Fictional customers: alice@example.com (premium, 2 orders) and bob@example.com (standard).")
    if st.button("↺  Start a new conversation", use_container_width=True):
        st.session_state.state = None
        st.session_state.history = []
        st.rerun()
    st.markdown("[View source on GitHub](https://github.com/Sami123d/nexus-support-extended)")

# --- MAIN ---
if not llm_provider.configured:
    st.warning("No AI key configured. Add `GEMINI_API_KEY = \"...\"` under Settings → Secrets "
               "(or set the GEMINI_API_KEY environment variable). Until then, messages go to the general specialist.")

active = (st.session_state.state or {}).get('active_agent') or 'supervisor'
active = {"general_support": "supervisor", "end": "supervisor"}.get(active, active)
cards = "".join(
    f'<div class="agent {"on" if key == active else ""}"><div class="ic">{ic}</div><div class="nm">{nm}</div><div class="ds">{ds}</div></div>'
    for key, ic, nm, ds in AGENTS
)
st.markdown(f"""
    <div class="hero-badge">⚡ LangGraph multi-agent · Live demo</div>
    <div class="hero-title">AI support that sends every ticket to <span>the right specialist</span></div>
    <div class="hero-sub">A supervisor model reads each message and hands it to an order, tech or billing agent,
    or escalates to a human. Card numbers, SSNs and emails are masked before anything reaches the model.</div>
    <div class="agents">{cards}</div>
""", unsafe_allow_html=True)

# Auto-initialize
if st.session_state.state is None:
    st.session_state.state = agent.start_conversation()
    greeting = st.session_state.state["messages"][-1].content
    st.session_state.history.append({"role": "assistant", "content": greeting})

# Human Takeover Logic
if st.session_state.state.get("is_human_takeover"):
    st.markdown("""
        <div class="paused"><b>A human specialist has taken over.</b> The AI is paused for this conversation.
        Start a new conversation from the sidebar to try again.</div>
    """, unsafe_allow_html=True)

# Display History
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Quick-start prompts (only before the first user message)
if len(st.session_state.history) <= 1 and not st.session_state.state.get("is_human_takeover"):
    st.caption("Try one of these:")
    cols = st.columns(len(DEMO_PROMPTS))
    for col, text in zip(cols, DEMO_PROMPTS):
        if col.button(text, use_container_width=True):
            st.session_state.pending_prompt = text
            st.rerun()

# Chat Input
typed = st.chat_input("Ask about an order, a login problem or a billing issue...",
                      disabled=bool(st.session_state.state.get("is_human_takeover")))
prompt = typed or st.session_state.pop("pending_prompt", None)

if prompt:
    # Append User Message
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Processing
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        # Stream from Graph
        for delta in agent.stream_message(st.session_state.state, prompt):
            for node, updated_state in delta.items():
                st.session_state.state = updated_state
                last_msg = updated_state["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    full_response = last_msg.content
                    placeholder.markdown(full_response + " ▌")
        
        placeholder.markdown(full_response)
        st.session_state.history.append({"role": "assistant", "content": full_response})
    
    st.rerun()
