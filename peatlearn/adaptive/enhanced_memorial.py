#!/usr/bin/env python3
"""
Enhanced Memorial Page for Dr. Ray Peat with beautiful design and technical details.
"""

import streamlit as st
from pathlib import Path

def render_enhanced_memorial():
    """Render an enhanced memorial page for Dr. Ray Peat with beautiful design and technical details."""
    
    # Custom CSS for memorial page
    st.markdown("""
    <style>
    .memorial-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .memorial-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 300;
    }
    
    .memorial-header .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-style: italic;
    }
    
    .quote-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        text-align: center;
        color: white;
        font-size: 1.3rem;
        font-style: italic;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    
    .principle-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .principle-card h4 {
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }
    
    .tech-section {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        border-left: 5px solid #667eea;
        color: #2c3e50;
    }
    
    .tech-section h3 {
        color: #1a365d;
        font-weight: bold;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        color: #333;
    }
    
    .feature-card h5 {
        color: #4CAF50;
        margin-bottom: 0.5rem;
    }
    
    .architecture-diagram {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        margin: 1rem 0;
        color: #2c3e50;
    }
    
    .architecture-diagram h3 {
        color: #1a365d;
        font-weight: bold;
    }
    
    .architecture-diagram p {
        color: #4a5568;
    }
    
    .memorial-footer {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Memorial Header
    st.markdown("""
    <div class="memorial-header">
        <h1>🕯️ In Memoriam: Dr. Raymond Peat</h1>
        <p class="subtitle">(1936–2022)</p>
        <p class="subtitle">Pioneering Bioenergetic Researcher and Independent Scholar</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Portrait and introduction
    col1, col2 = st.columns([1, 2])
    
    with col1:
        try:
            img_path = Path("data/assets/ray_peat.jpg")
            if img_path.exists():
                st.image(str(img_path), caption="Dr. Ray Peat", use_container_width=True)
            else:
                st.image("https://via.placeholder.com/300x400/667eea/white?text=Dr.+Ray+Peat", 
                        caption="Dr. Ray Peat", use_container_width=True)
        except Exception:
            st.image("https://via.placeholder.com/300x400/667eea/white?text=Dr.+Ray+Peat", 
                    caption="Dr. Ray Peat", use_container_width=True)
    
    with col2:
        st.markdown("""
        ### Remembering a Visionary
        
        Dr. Raymond Peat was a revolutionary thinker who fundamentally reshaped our understanding of 
        physiology, health, and the interconnected nature of biological systems. For over five decades, 
        he dedicated his life to unraveling the mysteries of cellular energy production and its profound 
        implications for human health.
        
        As an independent researcher with a PhD in Biology from the University of Oregon, Dr. Peat 
        challenged conventional medical paradigms with his groundbreaking **bioenergetic theory**. 
        His work bridged the gap between cutting-edge biochemistry and practical health applications.
        """)
    
    # Inspirational Quote
    st.markdown("""
    <div class="quote-container">
        "Energy and structure are interdependent at every level of organization."<br>
        <small>— Dr. Raymond Peat</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Core Principles Section
    st.markdown("## 🧬 The Bioenergetic Revolution: Core Principles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="principle-card">
            <h4>⚡ Energy as Central Variable</h4>
            <p>A cell's ability to produce and utilize energy determines its capacity to maintain structure, function, and resist stress.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="principle-card">
            <h4>🛡️ Protective Factors</h4>
            <p>Progesterone, DHEA, adequate carbohydrates, saturated fats, and essential minerals support optimal energy production.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="principle-card">
            <h4>🦋 Thyroid Connection</h4>
            <p>Thyroid hormones (T3/T4) optimize cellular machinery for maximum energy production with minimal waste.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="principle-card">
            <h4>💨 CO₂ Revelation</h4>
            <p>Carbon dioxide is not waste—it improves oxygen delivery, stabilizes enzymes, and protects cellular membranes.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # PeatLearn System Overview
    st.markdown("## 🎓 How PeatLearn Honors This Legacy")
    
    st.markdown("""
    <div class="tech-section">
        <h3>🚀 For Learners: Your Personal Ray Peat Mentor</h3>
        <div class="feature-grid">
            <div class="feature-card">
                <h5>🔍 Grounded Q&A System</h5>
                <p>Ask any question about metabolism, hormones, nutrition, or health and receive answers drawn directly from Ray Peat's original writings.</p>
            </div>
            <div class="feature-card">
                <h5>📚 Inline Citations & Sources</h5>
                <p>Every answer cites the passages it draws from, with the full source documents one click away for verification.</p>
            </div>
            <div class="feature-card">
                <h5>🛡️ Honest by Design</h5>
                <p>When the corpus doesn't support an answer, the system says so and abstains rather than improvising — a must for health-critical topics.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical Architecture Section
    st.markdown("---")
    st.markdown("# 🏗️ PeatLearn: Technical Architecture Deep Dive")
    
    st.markdown("""
    <div class="architecture-diagram">
        <h3 style="text-align: center; color: #667eea;">Grounded Retrieval-Augmented Q&A System</h3>
        <p style="text-align: center;">A multi-stage RAG pipeline built to keep every answer anchored in Ray Peat's corpus</p>
    </div>
    """, unsafe_allow_html=True)

    # System Architecture Visualization
    st.markdown("### 🎯 System Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🔍 Retrieval**
        - Pinecone Vector Database
        - HyDE query expansion
        - Two-pass semantic search
        - 552-document corpus
        """)

    with col2:
        st.markdown("""
        **🎯 Ranking**
        - Cohere rerank-4-pro
        - Cross-encoder fallback
        - MMR diversity selection
        - Source attribution
        """)

    with col3:
        st.markdown("""
        **🛡️ Grounding**
        - Confidence tiers
        - Entity-grounding checks
        - Grounding verifier
        - Abstains when unsupported
        """)
    
    # Technical Specifications
    st.markdown("### ⚙️ Core Technologies")
    
    tech_specs = {
        "Frontend": "Streamlit with custom CSS",
        "LLM": "Google Gemini 2.5 Flash / Flash Lite (Groq fallback)",
        "Embeddings": "Gemini embedding-001 (3072-dim)",
        "Reranker": "Cohere rerank-4-pro via OpenRouter (local cross-encoder fallback)",
        "Vector Search": "Pinecone (ray-peat-corpus-v3, 22,457 vectors)"
    }
    
    for tech, desc in tech_specs.items():
        st.markdown(f"**{tech}**: {desc}")
    
    # Benchmark
    st.markdown("### ⚡ Retrieval Quality")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("RAG Benchmark", "9.64/10", "30-question eval")
    with col2:
        st.metric("Source Diversity", "0.91", "across answers")
    with col3:
        st.metric("Avg Citations", "5.3", "per answer")

    # Future Enhancements
    st.markdown("### 🔮 Future Directions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🚀 Planned:**
        - Containerized deployment (Docker)
        - Hosted on a dedicated web domain
        - Conversation memory
        - Expanded corpus (forums, newsletters)
        """)

    with col2:
        st.markdown("""
        **🌟 Research Applications:**
        - Citation Network Analysis
        - Concept Evolution Tracking
        - Hypothesis Generation
        - Academic Collaboration
        """)
    
    # Memorial Footer
    st.markdown("""
    <div class="memorial-footer">
        <h3>Continuing Dr. Peat's Legacy</h3>
        <p>PeatLearn honors Dr. Peat's revolutionary insights by making his profound knowledge accessible to all who seek to understand the fundamental principles of health and human optimization.</p>
        <p><em>"The evidence from many fields of research is converging toward a recognition of the primacy of biological energy in health and disease."</em> — Dr. Raymond Peat</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Dr. Ray Peat Memorial - PeatLearn",
        page_icon="🕯️",
        layout="wide"
    )
    render_enhanced_memorial()
