# app.py
import streamlit as st
import pdfplumber
import docx2txt
from sentence_transformers import SentenceTransformer, util
import re
import time
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import hashlib
import json
from datetime import datetime, timedelta

# ---------- Configuration ----------
st.set_page_config(
    page_title="AI Resume Analyzer Pro", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'login_time' not in st.session_state:
    st.session_state.login_time = None

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .login-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    .login-form {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .keyword-tag {
        background: #f0f2f6;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .success-score {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .warning-score {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .user-info {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px 10px 0px 0px;
        color: white;
    }
    .signup-link {
        text-align: center;
        margin-top: 1rem;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Authentication Functions ----------
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from session state (in real app, use database)"""
    if 'users_db' not in st.session_state:
        # Default admin user and demo users
        st.session_state.users_db = {
            'admin': {
                'password': hash_password('admin123'),
                'email': 'admin@resumeanalyzer.com',
                'created_at': datetime.now().isoformat(),
                'role': 'admin'
            },
            'demo': {
                'password': hash_password('demo123'),
                'email': 'demo@example.com',
                'created_at': datetime.now().isoformat(),
                'role': 'user'
            }
        }
    return st.session_state.users_db

def save_user(username, password, email):
    """Save new user to database"""
    users_db = load_users()
    users_db[username] = {
        'password': hash_password(password),
        'email': email,
        'created_at': datetime.now().isoformat(),
        'role': 'user'
    }
    st.session_state.users_db = users_db

def verify_login(username, password):
    """Verify user login credentials"""
    users_db = load_users()
    if username in users_db:
        return users_db[username]['password'] == hash_password(password)
    return False

def is_session_valid():
    """Check if login session is still valid (24 hours)"""
    if st.session_state.login_time:
        session_duration = datetime.now() - st.session_state.login_time
        return session_duration < timedelta(hours=24)
    return False

# ---------- Login/Signup UI ----------
def show_login_page():
    """Display login and signup page"""
    st.markdown("""
    <div class="login-container">
        <h1>🚀 AI Resume Analyzer Pro</h1>
        <h3>Advanced AI-powered resume analysis with semantic matching</h3>
        <p>Please login to continue to your personalized dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for Login and Signup
    tab1, tab2 = st.tabs(["🔑 Login", "✨ Sign Up"])
    
    with tab1:
        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.subheader("Welcome Back!")
            
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            remember_me = st.checkbox("Remember me for 24 hours")
            
            submitted = st.form_submit_button("🚀 Login", use_container_width=True)
            
            if submitted:
                if not username or not password:
                    st.error("❌ Please fill in all fields")
                elif verify_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.login_time = datetime.now()
                    st.success("✅ Login successful! Redirecting...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        
        # Demo credentials info
        st.markdown("""
        <div class="signup-link">
            <h4>🎯 Demo Credentials</h4>
            <p><strong>Username:</strong> demo | <strong>Password:</strong> demo123</p>
            <p><strong>Admin:</strong> admin | <strong>Password:</strong> admin123</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        
        with st.form("signup_form"):
            st.subheader("Create New Account")
            
            new_username = st.text_input("Username", placeholder="Choose a username")
            new_email = st.text_input("Email", placeholder="Enter your email")
            new_password = st.text_input("Password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            
            agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            signup_submitted = st.form_submit_button("🎉 Create Account", use_container_width=True)
            
            if signup_submitted:
                users_db = load_users()
                
                if not all([new_username, new_email, new_password, confirm_password]):
                    st.error("❌ Please fill in all fields")
                elif new_username in users_db:
                    st.error("❌ Username already exists")
                elif new_password != confirm_password:
                    st.error("❌ Passwords don't match")
                elif len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters long")
                elif not agree_terms:
                    st.error("❌ Please agree to the Terms of Service")
                elif "@" not in new_email:
                    st.error("❌ Please enter a valid email address")
                else:
                    save_user(new_username, new_password, new_email)
                    st.success("🎉 Account created successfully! Please login.")
                    time.sleep(2)
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_user_dashboard():
    """Display user info and logout option"""
    users_db = load_users()
    user_info = users_db.get(st.session_state.username, {})
    
    with st.sidebar:
        st.markdown(f"""
        <div class="user-info">
            <h3>👋 Welcome, {st.session_state.username}!</h3>
            <p>📧 {user_info.get('email', 'N/A')}</p>
            <p>🕒 Login: {st.session_state.login_time.strftime('%H:%M') if st.session_state.login_time else 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Session info
        if st.session_state.login_time:
            session_duration = datetime.now() - st.session_state.login_time
            hours_left = 24 - session_duration.total_seconds() / 3600
            if hours_left > 0:
                st.info(f"⏰ Session expires in: {hours_left:.1f} hours")
            else:
                st.warning("⚠️ Session expired. Please login again.")
                logout()
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            logout()

def logout():
    """Clear session and logout user"""
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.login_time = None
    st.rerun()

# ---------- Load Models ----------
@st.cache_resource
def load_models():
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        nlp = None
    return sentence_model, nlp

model, nlp_model = load_models()

# ---------- Enhanced Helper Functions ----------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(uploaded_file):
    return docx2txt.process(uploaded_file)

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_advanced_keywords(text, method="tfidf", top_n=20):
    """Enhanced keyword extraction using multiple methods"""
    if method == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=top_n,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            return list(feature_names)
        except:
            # Fallback to simple method
            return extract_simple_keywords(text, top_n)
    elif method == "spacy" and nlp_model:
        doc = nlp_model(text)
        keywords = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "SKILL", "TECH"]:
                keywords.append(ent.text.lower())
        for token in doc:
            if token.pos_ in ["NOUN", "ADJ"] and len(token.text) > 3 and not token.is_stop:
                keywords.append(token.text.lower())
        return list(set(keywords))[:top_n]
    else:
        return extract_simple_keywords(text, top_n)

def extract_simple_keywords(text, top_n=20):
    words = [w.lower() for w in re.findall(r'\b\w+\b', text) if len(w) > 2]
    freq = Counter(words)
    return [k for k, v in freq.most_common(top_n)]

def highlight_keywords(resume, jd_keywords, missing_keywords):
    """Enhanced highlighting with different colors for found/missing keywords"""
    highlighted = resume
    
    # Highlight found keywords in green
    for kw in jd_keywords:
        if kw.lower() in resume.lower():
            highlighted = re.sub(rf"(?i)\b({re.escape(kw)})\b", r"🟢\1**", highlighted)
    
    return highlighted

def calculate_multiple_similarities(resume_text, jd_text):
    """Calculate similarity using multiple methods"""
    # Sentence Transformer similarity
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    semantic_similarity = util.cos_sim(resume_embedding, jd_embedding).item()
    
    # TF-IDF similarity
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        tfidf_similarity = semantic_similarity
    
    # Keyword overlap similarity
    resume_keywords = set(extract_simple_keywords(resume_text, 50))
    jd_keywords = set(extract_simple_keywords(jd_text, 50))
    keyword_similarity = len(resume_keywords.intersection(jd_keywords)) / len(jd_keywords.union(resume_keywords))
    
    return {
        'semantic': semantic_similarity,
        'tfidf': tfidf_similarity,
        'keyword': keyword_similarity,
        'overall': (semantic_similarity + tfidf_similarity + keyword_similarity) / 3
    }

def create_similarity_gauge(score):
    """Create an animated gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Match Score", 'font': {'size': 24}},
        delta={'reference': 70, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ff6b6b'},
                {'range': [50, 75], 'color': '#feca57'},
                {'range': [75, 100], 'color': '#48dbfb'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
    return fig

def analyze_resume_sections(resume_text):
    """Analyze different sections of the resume"""
    sections = {
        'Education': r'(education|degree|university|college|school|bachelor|master|phd)',
        'Experience': r'(experience|work|job|position|role|company|employer)',
        'Skills': r'(skills|technologies|tools|programming|software|technical)',
        'Projects': r'(projects|developed|built|created|implemented)',
        'Certifications': r'(certified|certification|license|credential)'
    }
    
    section_scores = {}
    for section, pattern in sections.items():
        matches = len(re.findall(pattern, resume_text, re.IGNORECASE))
        section_scores[section] = matches
    
    return section_scores

# ---------- Analysis Functions ----------
def quick_analysis(resume_text, jd_text, num_keywords):
    """Quick analysis mode"""
    with st.spinner("⚡ Quick analysis in progress..."):
        # Basic similarity
        similarities = calculate_multiple_similarities(resume_text, jd_text)
        overall_score = similarities['overall']
        
        # Keywords
        jd_keywords = extract_simple_keywords(jd_text, num_keywords)
        
        time.sleep(1)  # Animation effect
    
    # Results
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Score display
        score_percent = overall_score * 100
        if score_percent >= 75:
            st.markdown(f"""
            <div class="success-score">
                🎉 Excellent Match!<br>
                <span style="font-size: 2rem;">{score_percent:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        elif score_percent >= 50:
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%); color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                ⚠ Good Match<br>
                <span style="font-size: 2rem;">{score_percent:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 100%); color: #333; padding: 1rem; border-radius: 8px; text-align: center;">
                🔄 Needs Improvement<br>
                <span style="font-size: 2rem;">{score_percent:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Gauge chart
        fig = create_similarity_gauge(overall_score)
        st.plotly_chart(fig, use_container_width=True)
    
    # Key suggestions
    st.markdown("### 💡 Quick Recommendations")
    resume_keywords = set(extract_simple_keywords(resume_text, 50))
    missing_keywords = [kw for kw in jd_keywords if kw not in resume_keywords]
    
    if missing_keywords:
        st.warning(f"Consider adding these keywords: {', '.join(missing_keywords[:10])}")
    else:
        st.success("Great keyword coverage! 🎯")

def deep_analysis(resume_text, jd_text, keyword_method, num_keywords, threshold):
    """Deep analysis mode"""
    with st.spinner("🔬 Performing deep analysis..."):
        # Multiple similarity calculations
        similarities = calculate_multiple_similarities(resume_text, jd_text)
        
        # Advanced keyword extraction
        jd_keywords = extract_advanced_keywords(jd_text, keyword_method, num_keywords)
        resume_keywords = extract_advanced_keywords(resume_text, keyword_method, num_keywords)
        
        # Section analysis
        section_scores = analyze_resume_sections(resume_text)
        
        time.sleep(1.5)
    
    # Tabbed results
    tab1, tab2, tab3 = st.tabs(["📊 Similarity Analysis", "🔤 Keyword Analysis", "📋 Section Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Multi-method similarity
            sim_df = pd.DataFrame({
                'Method': ['Semantic', 'TF-IDF', 'Keyword', 'Overall'],
                'Score': [s*100 for s in [similarities['semantic'], similarities['tfidf'], 
                         similarities['keyword'], similarities['overall']]],
                'Color': ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
            })
            
            fig = px.bar(sim_df, x='Method', y='Score', color='Color',
                        title="Multi-Method Similarity Analysis",
                        color_discrete_map=dict(zip(sim_df.Color, sim_df.Color)))
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gauge for overall score
            fig_gauge = create_similarity_gauge(similarities['overall'])
            st.plotly_chart(fig_gauge, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Job Requirements Keywords")
            for kw in jd_keywords:
                if kw in [k.lower() for k in resume_keywords]:
                    st.markdown(f"✅ *{kw}*")
                else:
                    st.markdown(f"❌ {kw}")
        
        with col2:
            st.markdown("#### 📝 Resume Keywords")
            for kw in resume_keywords[:15]:
                st.markdown(f"🔸 {kw}")
        
        # Keyword overlap visualization
        overlap = len(set([k.lower() for k in jd_keywords]).intersection(set([k.lower() for k in resume_keywords])))
        overlap_percent = (overlap / len(jd_keywords)) * 100 if jd_keywords else 0
        
        st.metric("Keyword Match Rate", f"{overlap_percent:.1f}%", f"{overlap}/{len(jd_keywords)} keywords")
    
    with tab3:
        # Section analysis
        section_df = pd.DataFrame(list(section_scores.items()), columns=['Section', 'Mentions'])
        
        fig = px.horizontal_bar(section_df, x='Mentions', y='Section',
                               title="Resume Section Analysis",
                               color='Mentions',
                               color_continuous_scale='viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("#### 💡 Section Recommendations")
        for section, count in section_scores.items():
            if count < 2:
                st.warning(f"Consider strengthening the {section} section")
            else:
                st.success(f"{section} section looks good! ✓")

def expert_analysis(resume_text, jd_text, keyword_method, num_keywords, threshold):
    """Expert analysis mode with all features"""
    with st.spinner("🧠 Running expert-level analysis..."):
        # All calculations
        similarities = calculate_multiple_similarities(resume_text, jd_text)
        jd_keywords = extract_advanced_keywords(jd_text, keyword_method, num_keywords)
        resume_keywords = extract_advanced_keywords(resume_text, keyword_method, num_keywords)
        section_scores = analyze_resume_sections(resume_text)
        
        # Text statistics
        resume_stats = {
            'words': len(resume_text.split()),
            'characters': len(resume_text),
            'sentences': len(re.split(r'[.!?]+', resume_text))
        }
        
        time.sleep(2)
    
    # Comprehensive dashboard
    st.markdown("## 🎯 Expert Analysis Dashboard")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Match", f"{similarities['overall']*100:.1f}%")
    with col2:
        st.metric("Semantic Match", f"{similarities['semantic']*100:.1f}%")
    with col3:
        overlap = len(set([k.lower() for k in jd_keywords]).intersection(set([k.lower() for k in resume_keywords])))
        st.metric("Keyword Match", f"{(overlap/len(jd_keywords)*100):.1f}%")
    with col4:
        st.metric("Resume Length", f"{resume_stats['words']} words")
    
    # Detailed tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Analytics", "🔍 Keywords", "📄 Content", "🎯 Recommendations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Radar chart for different aspects
            categories = ['Semantic', 'TF-IDF', 'Keywords', 'Structure']
            values = [similarities['semantic']*100, similarities['tfidf']*100, 
                     (overlap/len(jd_keywords)*100), np.mean(list(section_scores.values()))*10]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Match Profile',
                line_color='rgb(67, 114, 238)'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                title="Multi-Dimensional Analysis",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Time-series style improvement suggestions
            improvement_data = {
                'Aspect': ['Keywords', 'Experience', 'Skills', 'Education', 'Format'],
                'Current': [overlap/len(jd_keywords)*100, section_scores.get('Experience', 0)*20, 
                           section_scores.get('Skills', 0)*20, section_scores.get('Education', 0)*20, 75],
                'Target': [90, 85, 80, 70, 90]
            }
            
            df = pd.DataFrame(improvement_data)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.Aspect, y=df.Current, mode='lines+markers', name='Current', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=df.Aspect, y=df.Target, mode='lines+markers', name='Target', line=dict(color='green')))
            fig.update_layout(title="Improvement Roadmap", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Advanced keyword analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Priority Keywords (Job Description)")
            priority_keywords = jd_keywords[:10]
            for i, kw in enumerate(priority_keywords):
                is_present = kw.lower() in [k.lower() for k in resume_keywords]
                status = "✅" if is_present else "❌"
                priority = "🔥" if i < 3 else "⭐" if i < 6 else ""
                st.markdown(f"{status} {priority} *{kw}*")
        
        with col2:
            st.markdown("#### 📊 Keyword Frequency Analysis")
            resume_word_freq = Counter(extract_simple_keywords(resume_text, 100))
            top_resume_words = dict(resume_word_freq.most_common(10))
            
            fig = px.bar(x=list(top_resume_words.keys()), y=list(top_resume_words.values()),
                        title="Top Resume Keywords")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Content analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Content Statistics")
            st.json({
                "Word Count": resume_stats['words'],
                "Character Count": resume_stats['characters'], 
                "Estimated Reading Time": f"{resume_stats['words']//200 + 1} min",
                "Keyword Density": f"{len(resume_keywords)/resume_stats['words']*100:.1f}%"
            })
        
        with col2:
            st.markdown("#### 🔍 Resume Preview (Highlighted)")
            # Show highlighted resume with limited length
            resume_keywords_lower = [k.lower() for k in resume_keywords]
            jd_keywords_in_resume = [k for k in jd_keywords if k.lower() in resume_keywords_lower]
            highlighted_resume = highlight_keywords(resume_text[:1000], jd_keywords_in_resume, [])
            st.markdown(highlighted_resume + "..." if len(resume_text) > 1000 else highlighted_resume)
    
    with tab4:
        # AI-powered recommendations
        st.markdown("#### 🎯 Personalized Recommendations")
        
        overall_score = similarities['overall'] * 100
        missing_keywords = [kw for kw in jd_keywords if kw.lower() not in [k.lower() for k in resume_keywords]]
        
        if overall_score < 60:
            st.error("🚨 *Major Improvements Needed*")
            st.markdown("*Priority Actions:*")
            st.markdown(f"1. Add these critical keywords: *{', '.join(missing_keywords[:5])}*")
            st.markdown("2. Restructure resume to better align with job requirements")
            st.markdown("3. Add more relevant experience examples")
        elif overall_score < 80:
            st.warning("⚠ *Good Progress - Fine-tuning Required*")
            st.markdown("*Suggested Improvements:*")
            st.markdown(f"1. Include missing keywords: *{', '.join(missing_keywords[:3])}*")
            st.markdown("2. Strengthen weak sections identified in analysis")
            st.markdown("3. Add quantifiable achievements")
        else:
            st.success("🎉 *Excellent Match - Ready to Submit!*")
            st.markdown("*Minor Enhancements:*")
            st.markdown("1. Perfect keyword alignment achieved")
            st.markdown("2. Consider adding specific metrics to achievements")
            st.markdown("3. Ensure formatting is ATS-friendly")
        
        # Action items
        st.markdown("#### ✅ Action Checklist")
        action_items = [
            f"Add missing keywords: {', '.join(missing_keywords[:3])}",
            "Quantify achievements with numbers",
            "Optimize resume length (1-2 pages)",
            "Use action verbs in experience descriptions",
            "Ensure consistent formatting"
        ]
        
        for item in action_items:
            st.checkbox(item, key=f"action_{hash(item)}")

# ---------- Main Application ----------
def main():
    # Check login status and session validity
    if not st.session_state.logged_in or not is_session_valid():
        if st.session_state.login_time and not is_session_valid():
            st.warning("⚠️ Your session has expired. Please login again.")
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.login_time = None
        show_login_page()
        return
    
    # Show user dashboard in sidebar
    show_user_dashboard()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI Resume Analyzer Pro</h1>
        <p>Advanced AI-powered resume analysis with semantic matching</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.title("⚙ Configuration")
    
    # Mode Toggle
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["🎯 Quick Match", "🔬 Deep Analysis", "📊 Expert Mode"],
        help="Choose your analysis depth"
    )
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        keyword_extraction = st.selectbox("Keyword Method", ["tfidf", "frequency", "spacy"])
        num_keywords = st.slider("Keywords to Extract", 10, 50, 25)
        similarity_threshold = st.slider("Match Threshold", 0.0, 1.0, 0.7)
    
    # Usage stats for logged-in user
    with st.sidebar.expander("📊 Your Usage Stats"):
        if 'user_analyses' not in st.session_state:
            st.session_state.user_analyses = 0
        
        st.metric("Analyses Today", st.session_state.user_analyses)
        st.progress(min(st.session_state.user_analyses / 10, 1.0))
        st.caption("Daily limit: 10 analyses")
    
    # File Uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Upload Resume")
        uploaded_resume = st.file_uploader(
            "Choose resume file", 
            type=['pdf', 'docx'],
            help="Upload your resume in PDF or DOCX format"
        )
    
    with col2:
        st.markdown("### 📋 Upload Job Description")
        uploaded_jd = st.file_uploader(
            "Choose job description file", 
            type=['pdf', 'docx'],
            help="Upload the job description in PDF or DOCX format"
        )
    
    # Alternative text input option
    if not uploaded_resume or not uploaded_jd:
        st.markdown("---")
        st.markdown("### ✍ Or Enter Text Directly")
        
        col1, col2 = st.columns(2)
        with col1:
            resume_text_input = st.text_area("Resume Text", height=300, placeholder="Paste your resume text here...")
        with col2:
            jd_text_input = st.text_area("Job Description Text", height=300, placeholder="Paste job description text here...")
        
        if resume_text_input and jd_text_input:
            # Use text inputs instead of uploaded files
            resume_text = clean_text(resume_text_input)
            jd_text = clean_text(jd_text_input)
            
            if st.button("🚀 Analyze Resume", use_container_width=True):
                # Check daily limit
                if st.session_state.user_analyses >= 10:
                    st.error("❌ Daily analysis limit reached. Please try again tomorrow or upgrade your plan.")
                    return
                
                st.session_state.user_analyses += 1
                
                # Run analysis based on selected mode
                if analysis_mode == "🎯 Quick Match":
                    quick_analysis(resume_text, jd_text, num_keywords)
                elif analysis_mode == "🔬 Deep Analysis":
                    deep_analysis(resume_text, jd_text, keyword_extraction, num_keywords, similarity_threshold)
                else:
                    expert_analysis(resume_text, jd_text, keyword_extraction, num_keywords, similarity_threshold)
    
    if uploaded_resume and uploaded_jd:
        # Check daily limit
        if st.session_state.user_analyses >= 10:
            st.error("❌ Daily analysis limit reached. Please try again tomorrow or upgrade your plan.")
            return
        
        # Process files
        with st.spinner("🔄 Processing documents..."):
            # Extract text
            resume_text = (extract_text_from_pdf(uploaded_resume) if uploaded_resume.type == "application/pdf" 
                          else extract_text_from_docx(uploaded_resume))
            jd_text = (extract_text_from_pdf(uploaded_jd) if uploaded_jd.type == "application/pdf" 
                      else extract_text_from_docx(uploaded_jd))
            
            resume_text = clean_text(resume_text)
            jd_text = clean_text(jd_text)
            
            if not resume_text or not jd_text:
                st.error("❌ Could not extract text from one or both files. Please check the file format.")
                return
            
            time.sleep(0.5)  # Smooth UX
        
        # Increment usage counter
        st.session_state.user_analyses += 1
        
        # Analysis based on mode
        if analysis_mode == "🎯 Quick Match":
            quick_analysis(resume_text, jd_text, num_keywords)
        elif analysis_mode == "🔬 Deep Analysis":
            deep_analysis(resume_text, jd_text, keyword_extraction, num_keywords, similarity_threshold)
        else:
            expert_analysis(resume_text, jd_text, keyword_extraction, num_keywords, similarity_threshold)
        
        # Show save/export options
        st.markdown("---")
        st.markdown("### 💾 Save & Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Save Analysis"):
                # In real app, save to database
                st.success("✅ Analysis saved to your dashboard!")
        
        with col2:
            if st.button("📄 Export PDF Report"):
                st.info("🔄 PDF export feature coming soon!")
        
        with col3:
            if st.button("📧 Email Report"):
                st.info("📧 Email feature coming soon!")

# ---------- Footer ----------
def show_footer():
    """Display app footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🚀 <strong>AI Resume Analyzer Pro</strong> | Made with ❤️ using Streamlit</p>
        <p>© 2024 Resume Analyzer Pro. All rights reserved.</p>
        <p>
            <a href="#" style="color: #667eea; text-decoration: none;">Privacy Policy</a> | 
            <a href="#" style="color: #667eea; text-decoration: none;">Terms of Service</a> | 
            <a href="#" style="color: #667eea; text-decoration: none;">Support</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------- Run Application ----------
if __name__ == "__main__":
    main()
    
    # Show footer only if logged in
    if st.session_state.logged_in:
        show_footer()