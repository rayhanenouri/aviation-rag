import streamlit as st
import plotly.graph_objects as go
import time
from pathlib import Path


st.set_page_config(
    page_title="Aviation Maintenance Intelligence",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ife display inspired design system with jetbrains mono
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    * {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background-color: #08090f;
    }

    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .stMarkdown, .stText, p, span, div, label {
        color: #e0e8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'JetBrains Mono', monospace !important;
        color: #e0e8f0 !important;
        letter-spacing: 0.08em;
    }

    .header-container {
        border-top: 1px solid #00b4d8;
        border-bottom: 1px solid #1a2744;
        padding: 20px 0;
        margin-bottom: 40px;
    }

    .header-title {
        font-size: 13px;
        letter-spacing: 0.3em;
        color: #6b8caa;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .header-stats {
        font-size: 10px;
        color: #6b8caa;
        letter-spacing: 0.08em;
        font-weight: 400;
    }

    .section-label {
        font-size: 10px;
        letter-spacing: 0.15em;
        color: #6b8caa;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 12px;
        margin-top: 32px;
    }

    .stTextArea textarea {
        background-color: #0d1117 !important;
        color: #e0e8f0 !important;
        border: 1px solid #1a2744 !important;
        border-radius: 0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
        line-height: 1.6 !important;
        padding: 12px !important;
    }

    .stTextArea textarea:focus {
        border-color: #00b4d8 !important;
        box-shadow: none !important;
        outline: none !important;
    }

    .stTextArea textarea::placeholder {
        color: #1a2744 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stTextArea label {
        display: none;
    }

    .stButton button {
        background-color: #00b4d8 !important;
        color: #08090f !important;
        border: none !important;
        border-radius: 0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        width: 100% !important;
        padding: 14px 0 !important;
        transition: background-color 0.2s ease !important;
    }

    .stButton button:hover {
        background-color: #0077a8 !important;
        border: none !important;
    }

    .answer-card {
        background-color: #0d1117;
        border-left: 2px solid #00b4d8;
        padding: 16px;
        margin-top: 12px;
        margin-bottom: 32px;
    }

    .answer-card p {
        font-size: 13px;
        line-height: 1.7;
        color: #e0e8f0;
        margin: 0;
    }

    div[data-testid="stExpander"] {
        background-color: #0d1117 !important;
        border: 1px solid #1a2744 !important;
        border-radius: 0 !important;
        margin-bottom: 8px !important;
    }

    div[data-testid="stExpander"] summary {
        color: #e0e8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        padding: 12px !important;
    }

    div[data-testid="stExpander"] summary:hover {
        background-color: #1a2744 !important;
    }

    div[data-testid="stCodeBlock"] {
        background-color: #08090f !important;
        border-left: 1px solid #1a2744 !important;
        margin-top: 8px !important;
    }

    div[data-testid="stCodeBlock"] code {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        line-height: 1.6 !important;
        color: #e0e8f0 !important;
    }

    .stCaptionContainer p {
        font-size: 9px !important;
        color: #6b8caa !important;
        letter-spacing: 0.1em !important;
        margin-bottom: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 500 !important;
    }

    .stProgress > div > div {
        background-color: #1a2744 !important;
        height: 2px !important;
    }

    .stProgress > div > div > div {
        background-color: #00b4d8 !important;
    }

    .stProgress {
        margin-bottom: 16px !important;
    }

    .footer-container {
        border-top: 1px solid #1a2744;
        padding-top: 24px;
        margin-top: 48px;
        text-align: center;
    }

    .footer-text {
        font-size: 10px;
        color: #1a4060;
        letter-spacing: 0.05em;
        line-height: 1.8;
    }

    section[data-testid="stSidebar"] {
        display: none !important;
    }

    .stAlert {
        background-color: #0d1117 !important;
        color: #e0e8f0 !important;
        border: 1px solid #1a2744 !important;
        border-radius: 0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
    }

    .processing-container {
        width: 100%;
        height: 3px;
        background-color: #0d1117;
        position: relative;
        overflow: hidden;
        margin: 24px 0;
    }

    @keyframes pulse {
        0% {
            left: -40%;
            width: 40%;
        }
        100% {
            left: 100%;
            width: 40%;
        }
    }

    .processing-pulse {
        position: absolute;
        height: 100%;
        background: linear-gradient(90deg, transparent, #00b4d8, transparent);
        animation: pulse 1.5s ease-in-out infinite;
    }

    .processing-text {
        font-size: 10px;
        color: #6b8caa;
        letter-spacing: 0.1em;
        text-align: center;
        margin-top: 8px;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """cache rag pipeline to avoid reloading on every interaction"""
    import time
    from src.vector_store import AviationVectorStore
    from src.rag_pipeline import AviationRAG

    start = time.time()
    print("\n" + "="*60)
    print("INITIALIZING AVIATION MAINTENANCE INTELLIGENCE...")
    print("="*60)

    vs = AviationVectorStore()
    vs.load_existing()
    rag = AviationRAG(vs)

    elapsed = time.time() - start
    print(f"✓ Pipeline loaded in {elapsed:.2f}s")
    print("="*60 + "\n")

    return rag


def get_system_status(vector_store):
    """gather system status metrics for header display"""
    try:
        chunk_count = vector_store.vector_store._collection.count()
        all_data = vector_store.vector_store.get()
        unique_files = set()
        for metadata in all_data['metadatas']:
            unique_files.add(metadata['source_file'])

        return {
            'chunk_count': chunk_count,
            'document_count': len(unique_files),
            'embedding_model': 'all-MiniLM-L6-v2',
            'llm_model': 'gpt-oss-120b (groq)'
        }
    except Exception as e:
        return {
            'chunk_count': 0,
            'document_count': 0,
            'embedding_model': 'N/A',
            'llm_model': 'N/A'
        }


def create_radar_chart(sources):
    """create radar chart showing similarity scores across source documents"""
    # prepare data for radar chart
    categories = []
    scores = []

    for i, source in enumerate(sources, 1):
        # shorten filename for readability on radar axes
        filename = source['file'].replace('.pdf', '')
        if len(filename) > 15:
            filename = filename[:12] + '...'
        label = f"{filename} p{source['page']}"
        categories.append(label)
        scores.append(source['similarity_score'])

    # close the radar loop
    categories_loop = categories + [categories[0]]
    scores_loop = scores + [scores[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=scores_loop,
        theta=categories_loop,
        fill='toself',
        fillcolor='rgba(0, 180, 216, 0.15)',
        line=dict(color='#00b4d8', width=2),
        marker=dict(color='#00b4d8', size=6),
        name='Similarity'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                tickfont=dict(size=9, color='#6b8caa', family='JetBrains Mono'),
                gridcolor='#1a2744',
                gridwidth=1
            ),
            angularaxis=dict(
                tickfont=dict(size=9, color='#e0e8f0', family='JetBrains Mono'),
                gridcolor='#1a2744',
                gridwidth=1
            ),
            bgcolor='#08090f'
        ),
        showlegend=False,
        title=dict(
            text='SOURCE RELEVANCE ANALYSIS',
            font=dict(size=10, color='#6b8caa', family='JetBrains Mono'),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        paper_bgcolor='#08090f',
        plot_bgcolor='#08090f',
        margin=dict(l=60, r=60, t=60, b=60),
        height=400
    )

    return fig


# load pipeline once on startup with loading indicator
with st.spinner("INITIALIZING AVIATION MAINTENANCE INTELLIGENCE..."):
    rag = load_pipeline()
    status = get_system_status(rag.vector_store)

# header section with ife display style
st.markdown(f"""
<div class="header-container">
    <div class="header-title">AVIATION MAINTENANCE INTELLIGENCE</div>
    <div class="header-stats">{status['chunk_count']} CHUNKS INDEXED  |  {status['llm_model']}  |  {status['embedding_model']}</div>
</div>
""", unsafe_allow_html=True)

# query section
st.markdown('<div class="section-label">QUERY</div>', unsafe_allow_html=True)

query_text = st.text_area(
    "query_input",
    height=120,
    placeholder="enter maintenance query...",
    label_visibility="collapsed"
)

submit_button = st.button("Submit Query")

# handle query submission
if submit_button and query_text:
    # custom processing indicator instead of default spinner
    processing_placeholder = st.empty()
    with processing_placeholder.container():
        st.markdown("""
        <div class="processing-container">
            <div class="processing-pulse"></div>
        </div>
        <div class="processing-text">PROCESSING QUERY</div>
        """, unsafe_allow_html=True)

    # execute query
    result = rag.query(query_text)

    # debug: print result structure to terminal
    print(f"\n=== DEBUG: Result keys: {result.keys()}")
    print(f"=== DEBUG: Full result: {result}")

    # clear processing indicator
    processing_placeholder.empty()

    # display radar chart
    if result['sources']:
        st.markdown('<div style="margin-top: 32px; margin-bottom: 32px;">', unsafe_allow_html=True)
        radar_chart = create_radar_chart(result['sources'])
        st.plotly_chart(radar_chart, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # display answer
    st.markdown('<div class="section-label">RESPONSE</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="answer-card">
        <p>{result['answer']}</p>
    </div>
    """, unsafe_allow_html=True)

    # display sources
    st.markdown('<div class="section-label">SOURCE DOCUMENTS</div>', unsafe_allow_html=True)

    for i, source in enumerate(result['sources']):
        # plain container without expander to avoid arrow icons
        with st.container():
            # source header
            label = f"[{i+1}]  {source['file']}  ·  PAGE {source['page']}  ·  {source['document_type']}"
            st.markdown(f"""
            <div style="background-color: #0d1117; border: 1px solid #1a2744; padding: 12px; margin-bottom: 8px;">
                <div style="color: #e0e8f0; font-size: 11px; font-family: 'JetBrains Mono', monospace;">
                    {label}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # relevance indicator bar
            score = source['similarity_score']
            st.caption(f"RELEVANCE: {score:.0f}%")
            st.progress(score / 100.0)

            # source content as plain readable text
            st.text(source['content'])

elif submit_button and not query_text:
    st.warning("query field required")

# footer
st.markdown("""
<div class="footer-container">
    <div class="footer-text">
        AVIATION MAINTENANCE INTELLIGENCE<br>
        LangChain · ChromaDB · Sentence Transformers · Groq<br>
        Rayhane Nouri · ENSIT Tunisia
    </div>
</div>
""", unsafe_allow_html=True)
