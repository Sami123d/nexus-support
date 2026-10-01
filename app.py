# app.py

import streamlit as st
import time
import os
import re
from datetime import datetime
from customer_support_agent import CustomerSupportAgent, llm_provider
from langchain_core.messages import HumanMessage, AIMessage

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Agentic Support | Enterprise",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM DESIGN SYSTEM (CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');

    :root {
        --primary: #6366f1;
        --primary-hover: #4f46e5;
        --bg-main: #0b0f1a;
        --bg-sidebar: #0f172a;
        --glass-bg: rgba(30, 41, 59, 0.7);
        --glass-border: rgba(255, 255, 255, 0.1);
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
    }

    /* Global Overrides */
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #0b0f1a);
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar);
        border-right: 1px solid var(--glass-border);
    }

    [data-testid="stSidebarNav"] {
        display: none;
    }

    .sidebar-header {
        padding: 1.5rem 0;
        text-align: center;
        border-bottom: 1px solid var(--glass-border);
        margin-bottom: 2rem;
    }

    /* Glass Cards */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }

    /* Chat Styling */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 0 !important;
    }

    /* User Message Bubble */
    [data-testid="stChatMessage"]:nth-child(even) .stChatMessageContent {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: white !important;
        border-radius: 20px 20px 4px 20px !important;
        padding: 1rem 1.25rem !important;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3);
    }

    /* AI Message Bubble */
    [data-testid="stChatMessage"]:nth-child(odd) .stChatMessageContent {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid var(--glass-border) !important;
        color: var(--text-main) !important;
        border-radius: 20px 20px 20px 4px !important;
        padding: 1rem 1.25rem !important;
    }

    /* Custom Input */
    .stChatInputContainer {
        padding: 1.5rem 0 !important;
    }

    button[data-testid="stChatInputButton"] {
        background-color: var(--primary) !important;
        border-radius: 10px !important;
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .status-online { background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.2); }
    .status-active { background: rgba(99, 102, 241, 0.1); color: #6366f1; border: 1px solid rgba(99, 102, 241, 0.2); }

    /* Hide standard Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stChatMessage { animation: fadeIn 0.5s ease forwards; }

</style>
""", unsafe_allow_html=True)

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

# --- SIDEBAR (Premium Console) ---
with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            <h2 style='margin:0; color:#6366f1;'>PREMIUM</h2>
            <p style='color:#94a3b8; font-size:0.8rem; margin:0;'>Agentic Support Cloud</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.state:
        s = st.session_state.state
        
        st.markdown("<h4 style='color:#f8fafc; font-size:0.9rem; margin-bottom:1rem;'>LIVE SESSION METRICS</h4>", unsafe_allow_html=True)
        
        # Agent Card
        agent_name = s.get('active_agent', 'Supervisor').replace('_',' ').title()
        st.markdown(f"""
            <div class="glass-card">
                <small style='color:#94a3b8; display:block; margin-bottom:4px;'>ACTIVE OPERATOR</small>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='font-weight:600; color:#fff;'>{agent_name}</span>
                    <span class="status-badge status-active">Online</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Stats Cards
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="glass-card" style="text-align:center;">
                    <small style='color:#94a3b8;'>TIER</small><br/>
                    <span style='color:#6366f1; font-weight:700;'>{s.get("customer_tier", "Standard").upper()}</span>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="glass-card" style="text-align:center;">
                    <small style='color:#94a3b8;'>TOKENS</small><br/>
                    <span style='color:#fff; font-weight:700;'>{int(s.get('total_tokens', 0))}</span>
                </div>
            """, unsafe_allow_html=True)

        # Cost Analysis
        cost = s.get('total_tokens', 0) * 0.00000014
        st.markdown(f"""
            <div class="glass-card">
                <div style='display:flex; justify-content:space-between;'>
                    <small style='color:#94a3b8;'>ELAPSED COST</small>
                    <b style='color:#10b981;'>${cost:.6f}</b>
                </div>
                <div style='height:4px; background:rgba(255,255,255,0.05); border-radius:10px; margin-top:10px;'>
                    <div style='width:{min(cost*10000, 100)}%; height:100%; background:#10b981; border-radius:10px;'></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h4 style='color:#f8fafc; font-size:0.9rem; margin-top:2rem; margin-bottom:1rem;'>INFRASTRUCTURE</h4>", unsafe_allow_html=True)
        engine = getattr(llm_provider.primary, "model_name", "not configured")
        db_kind = "Postgres" if os.getenv("DATABASE_URL", "").startswith("postgres") else "SQLite"
        st.markdown(f'<div class="status-badge status-online" style="width:100%; margin-bottom:0.5rem;">Engine: {engine}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-badge status-online" style="width:100%; margin-bottom:0.5rem;">DB: {db_kind}</div>', unsafe_allow_html=True)
        st.markdown('<div class="status-badge status-online" style="width:100%;">Security: PII Masking</div>', unsafe_allow_html=True)

    st.markdown("<h4 style='color:#f8fafc; font-size:0.9rem; margin-top:2rem;'>TRY THE DEMO</h4>", unsafe_allow_html=True)
    st.markdown(
        "- `My email is alice@example.com`\n"
        "- `Where is my order?`\n"
        "- `I can't log in to my account`\n"
        "- `I want a refund for a double charge`"
    )
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    if st.button("RESET SESSION", use_container_width=True, type="secondary"):
        st.session_state.state = None
        st.session_state.history = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
if not llm_provider.configured:
    st.warning("No AI key configured. Add `GEMINI_API_KEY = \"...\"` under Settings → Secrets "
               "(or set the GEMINI_API_KEY environment variable). Until then, messages go to the general specialist.")

st.markdown("""
    <div style='margin-top: 2rem; margin-bottom: 2rem;'>
        <h1 style='font-size: 2.5rem; margin-bottom: 0px;'>Enterprise Intelligence</h1>
        <p style='color: #94a3b8; font-size: 1.1rem;'>Multi-Agent Orchestration & Support Terminal</p>
    </div>
""", unsafe_allow_html=True)

# Auto-initialize
if st.session_state.state is None:
    st.session_state.state = agent.start_conversation()
    greeting = st.session_state.state["messages"][-1].content
    st.session_state.history.append({"role": "assistant", "content": greeting})

# Human Takeover Logic
if st.session_state.state.get("is_human_takeover"):
    st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); padding: 1.5rem; border-radius: 12px; color: #fca5a5; text-align: center; margin-bottom: 2rem;">
            <h3 style='margin:0; color:#ef4444;'>AI SESSION PAUSED</h3>
            <p style='margin:5px 0 0 0;'>A human specialist has joined the conversation. Input is disabled.</p>
        </div>
    """, unsafe_allow_html=True)

# Display History
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
prompt = st.chat_input("Input command or query...", disabled=bool(st.session_state.state.get("is_human_takeover")))

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
