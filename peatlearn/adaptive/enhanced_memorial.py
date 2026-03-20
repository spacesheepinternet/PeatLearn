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
                <h5>🔍 Intelligent Q&A System</h5>
                <p>Ask any question about metabolism, hormones, nutrition, or health and receive comprehensive answers backed by Ray Peat's original writings.</p>
            </div>
            <div class="feature-card">
                <h5>🎯 Adaptive Learning Experience</h5>
                <p>Take personalized quizzes that adjust to your knowledge level and build mastery progressively.</p>
            </div>
            <div class="feature-card">
                <h5>📊 Progress Analytics</h5>
                <p>Monitor your learning journey with detailed analytics and visualize your growing expertise.</p>
            </div>
            <div class="feature-card">
                <h5>🧠 AI-Powered Insights</h5>
                <p>Get recommendations and explanations powered by advanced machine learning algorithms.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical Architecture Section
    st.markdown("---")
    st.markdown("# 🏗️ PeatLearn: Technical Architecture Deep Dive")
    
    st.markdown("""
    <div class="architecture-diagram">
        <h3 style="text-align: center; color: #667eea;">Advanced AI-Powered Learning Platform</h3>
        <p style="text-align: center;">State-of-the-art implementation of modern AI techniques for education</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Architecture Visualization
    st.markdown("### 🎯 System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔍 RAG System**
        - Pinecone Vector Database
        - Semantic Search
        - Source Attribution
        - <500ms Response Time
        """)
    
    with col2:
        st.markdown("""
        **🧠 Adaptive Learning**
        - Item Response Theory
        - Real-time Ability Updates
        - Dynamic Difficulty
        - Personalized Quizzes
        """)
    
    with col3:
        st.markdown("""
        **📊 Analytics Engine**
        - Learning Trajectories
        - Performance Metrics
        - Predictive Modeling
        - Recommendation System
        """)
    
    # Technical Specifications
    st.markdown("### ⚙️ Core Technologies")
    
    tech_specs = {
        "Backend Framework": "FastAPI (async, high-performance)",
        "AI/ML Stack": "Gemini 2.5 Flash Lite, Pinecone, Advanced Embeddings",
        "Frontend": "Streamlit with Custom CSS/JS",
        "Database": "SQLite (user data) + Pinecone (vector search)",
        "Deployment": "Docker containers, Microservice architecture"
    }
    
    for tech, desc in tech_specs.items():
        st.markdown(f"**{tech}**: {desc}")
    
    # Performance Metrics
    st.markdown("### ⚡ Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("RAG Queries", "<500ms", "avg response")
    with col2:
        st.metric("Quiz Generation", "<1.5s", "complete set")
    with col3:
        st.metric("Vector Search", "<100ms", "semantic retrieval")
    with col4:
        st.metric("Analytics", "<200ms", "dashboard updates")
    
    # Adaptive Learning Mathematics
    st.markdown("### 📈 Adaptive Learning Model")
    
    st.markdown("""
    **Item Response Theory Implementation:**
    
    ```python
    # Expected probability of correct response
    P(θ, b) = 1 / (1 + exp(-1.7 * (θ - b)))
    
    # Ability update after response
    θ_new = θ + K_θ * (observed - expected)
    
    # Item difficulty update
    b_new = b + K_b * (expected - observed)
    ```
    
    Where:
    - **θ (theta)**: User ability parameter
    - **b**: Item difficulty parameter  
    - **K_θ, K_b**: Learning rates (0.15, 0.07 respectively)
    """)
    
    # API Architecture
    st.markdown("### 🔌 API Architecture")
    
    st.markdown("""
    **Service Endpoints:**
    
    | Service | Port | Purpose |
    |---------|------|---------|
    | RAG API | 8000 | Q&A, Search, Citations |
    | Advanced ML | 8001 | Adaptive Learning, Quiz Generation |
    | Streamlit UI | 8501 | User Interface |
    | Analytics | 8002 | Performance Metrics |
    """)
    
    # Future Enhancements
    st.markdown("### 🔮 Future Enhancements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🚀 Planned Features:**
        - Multimodal Learning (visual/audio)
        - Conversation Memory
        - Interactive Simulations
        - Cross-platform Sync
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
