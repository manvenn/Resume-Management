import streamlit as st
import re
import time
import os
from io import BytesIO
from PyPDF2 import PdfReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ---------------------------------------------------------
# Page Configuration (MUST be the first Streamlit command)
# ---------------------------------------------------------
st.set_page_config(
    page_title="ResuMatch AI | Intelligent Talent & Resume Lifecycle Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "optimized_resume" not in st.session_state:
    st.session_state.optimized_resume = ""
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "sample_mode" not in st.session_state:
    st.session_state.sample_mode = False

# ---------------------------------------------------------
# Groq Client Initialization
# ---------------------------------------------------------
GROQ_API_KEY = "gsk_"
client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------
# Skills Database
# ---------------------------------------------------------
SKILLS_DATABASE = [
    # Computer Science & Software
    "python", "java", "c++", "sql", "machine learning", "artificial intelligence",
    "data structures", "algorithms", "git", "github", "html", "css", "javascript",
    # ECE Skills
    "embedded systems", "electronics", "microcontrollers", "communication systems",
    "signal processing", "pcb design", "vlsi",
    # Civil Engineering Skills
    "autocad", "revit", "qgis", "staad pro", "surveying",
    "construction management", "structural analysis", "building design"
]

SAMPLE_RESUME_TEXT = """ALEX MORGAN
Software Engineer | alex.morgan@email.com | GitHub: github.com/alexmorgan

PROFESSIONAL SUMMARY
Versatile Software Engineer with 3+ years of experience developing scalable web applications and automated data pipelines. Proficient in Python, SQL, JavaScript, HTML, CSS, Git, and Machine Learning algorithms.

TECHNICAL SKILLS
- Languages: Python, Java, JavaScript, SQL, HTML, CSS
- Frameworks & Tools: Git, GitHub, Machine Learning, Data Structures, Algorithms, Artificial Intelligence

EXPERIENCE
Software Engineer - Tech Solutions Inc (2022 - Present)
- Built full-stack web applications using Python, SQL, and modern frontend technologies.
- Implemented automated ML pipelines and data processing algorithms improving data throughput by 35%.
- Collaborated in an agile engineering team using Git/GitHub for version control.

EDUCATION
B.S. in Computer Science - State University (2018 - 2022)
"""

SAMPLE_JOB_DESCRIPTION = """We are looking for a Senior Python & Machine Learning Engineer to join our AI solutions team.

RESPONSIBILITIES:
- Design and build intelligent software applications utilizing Machine Learning, Artificial Intelligence, and Signal Processing.
- Write clean, maintainable Python and C++ code for high-throughput production environments.
- Work with SQL databases and optimize complex data structures and algorithms.
- Collaborate on embedded systems and communication systems integrations.

REQUIREMENTS:
- Strong proficiency in Python, C++, SQL, Machine Learning, and Artificial Intelligence.
- Experience with Git, Data Structures, Algorithms, Signal Processing, and Embedded Systems.
- Strong communication and analytical problem-solving skills.
"""

# ---------------------------------------------------------
# Custom High-End Modern Styling (Glassmorphism & SaaS Design)
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Global Font & Canvas Reset */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
    color: #0f172a;
}

/* Sidebar Modern Dark Theme */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8 !important;
}

/* Hero Header Banner */
.hero-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
    border-radius: 20px;
    padding: 32px 36px;
    color: #ffffff;
    margin-bottom: 24px;
    box-shadow: 0 20px 25px -5px rgba(15, 23, 42, 0.25), 0 8px 10px -6px rgba(15, 23, 42, 0.1);
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
    border: 1px solid rgba(168, 85, 247, 0.4);
    color: #c084fc;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    margin: 0 0 8px 0 !important;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    margin: 0;
    max-width: 720px;
    line-height: 1.5;
}

/* Glassmorphic Metric Cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04), 0 4px 6px -4px rgba(0, 0, 0, 0.02);
    transition: all 0.2s ease-in-out;
    height: 100%;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.07), 0 8px 10px -6px rgba(0, 0, 0, 0.03);
}
.metric-label {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.02em;
}
.metric-value.emerald { color: #059669; }
.metric-value.indigo { color: #4f46e5; }
.metric-value.amber { color: #d97706; }
.metric-value.rose { color: #e11d48; }

/* Custom Skill Badges / Chips */
.skills-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}
.skill-chip {
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: transform 0.15s ease;
}
.skill-chip:hover {
    transform: scale(1.04);
}
.skill-chip.matched {
    background: rgba(16, 185, 129, 0.12);
    color: #047857;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.skill-chip.missing {
    background: rgba(239, 68, 68, 0.12);
    color: #b91c1c;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Quote / Recruiter Feedback Card */
.feedback-box {
    background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
    border-left: 4px solid #3b82f6;
    border-radius: 12px;
    padding: 20px;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #1e293b;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

/* Role Suitability Badge Card */
.role-card {
    background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
    border: 1px solid #e9d5ff;
    border-radius: 16px;
    padding: 24px;
}
.role-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #6b21a8;
    margin-bottom: 8px;
}

/* Custom Styled Streamlit Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    height: 52px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3), 0 4px 6px -4px rgba(79, 70, 229, 0.2) !important;
    transition: all 0.2s ease-in-out !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 20px 25px -5px rgba(79, 70, 229, 0.4), 0 8px 10px -6px rgba(79, 70, 229, 0.3) !important;
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
}

/* Streamlit Tabs Styling */
button[data-baseweb="tab"] {
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 12px 20px !important;
    border-radius: 8px !important;
}
button[aria-selected="true"] {
    color: #4f46e5 !important;
}

/* File Uploader & Inputs */
.stTextArea textarea {
    border-radius: 12px !important;
    border: 1px solid #cbd5e1 !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
}
div[data-testid="stFileUploader"] {
    border-radius: 12px !important;
}

/* Subheaders & Labels */
h2, h3 {
    color: #0f172a;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Component
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <span style="font-size: 2.5rem;">📄</span>
        <h2 style="margin: 8px 0 0 0; font-weight: 800; font-size: 1.4rem; color: #ffffff !important;">ResuMatch AI</h2>
        <p style="font-size: 0.8rem; color: #94a3b8 !important;">Intelligent Talent Lifecycle Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📌 Key Features")
    st.markdown("""
    - 🎯 **NLP Content Similarity** (TF-IDF)
    - ⚡ **Automated Skill Gap Analysis**
    - 💬 **Recruiter AI Feedback**
    - 🚀 **Target Career Role Matching**
    - 📝 **ATS Resume Optimization & PDF Export**
    """)

    st.divider()

    st.markdown("### 🛠️ Database Taxonomy")
    with st.expander("View Skills Database", expanded=False):
        st.write(", ".join([s.title() for s in SKILLS_DATABASE]))

    st.markdown("""
    <div style="margin-top: 40px; padding: 12px; background: rgba(255,255,255,0.05); border-radius: 10px; font-size: 0.75rem; text-align: center;">
        Powered by Groq LLM Engine & Scikit-Learn NLP
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Page Header (Hero)
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">⚡ GROQ LLM & NLP ATS ENGINE</div>
    <h1 class="hero-title">Intelligent Resume Lifecycle Management</h1>
    <p class="hero-subtitle">Upload your resume and job target to analyze keyword alignment, identify skill gaps, extract recruiter feedback, and generate ATS-optimized PDF resumes.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Core Helper Functions
# ---------------------------------------------------------
def extract_resume_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def calculate_match_score(resume_text, job_description):
    documents = [resume_text, job_description]
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(float(similarity_score[0][0]) * 100, 2)

def extract_skills(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    found_skills = []
    for skill in SKILLS_DATABASE:
        skill_words = skill.split()
        if len(skill_words) == 1:
            if skill in words:
                found_skills.append(skill)
        else:
            if skill in text:
                found_skills.append(skill)
    return sorted(list(set(found_skills)))

def calculate_skill_score(resume_skills, job_skills):
    if not job_skills:
        return 0.0
    matched_count = len(set(resume_skills) & set(job_skills))
    skill_score = (matched_count / len(job_skills)) * 100
    return round(skill_score, 2)

def generate_feedback(resume_text, job_description, match_score, missing_skills, suggested_role):
    prompt = f"""
    You are an expert AI recruitment assistant and talent assessor.

    Resume Overall Match Percentage: {match_score}%
    Suggested Career Path: {suggested_role}
    Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}

    Resume Snippet:
    {resume_text[:3000]}

    Job Description Snippet:
    {job_description[:2000]}

    Generate a concise, professional recruiter executive evaluation in 3–5 lines using a polished third-person tone.
    Highlight:
    1. Overall alignment score summary
    2. Key strengths demonstrated
    3. Critical missing skill gaps
    4. Actionable resume improvement suggestions
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Recruiter feedback error: {str(e)}"

def suggest_role(resume_text):
    prompt = f"""
    You are a career recommendation AI.
    Analyze this resume text and determine:
    1. The BEST suited career role
    2. A brief 1–2 sentence rationale.

    Format EXACTLY as:
    Career Role: [Role Title]
    Why: [Short professional explanation in third-person tone]

    Do NOT use "you" or "your".

    Resume:
    {resume_text[:3000]}
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "General Engineering Candidate\n\nCould not generate AI recommendation."

def generate_optimized_resume(resume_text, job_description):
    prompt = f"""
    You are an expert ATS resume optimizer.

    Rewrite and optimize the resume below to best align with the target job description.

    RULES:
    - Keep ALL information truthful; do NOT invent fake projects, experience, or skills.
    - Improve bullet points with strong action verbs and professional phrasing.
    - Naturally integrate relevant job keywords where applicable.
    - Reorder skills by job relevance.
    - Use clear markdown sections (e.g. PROFESSIONAL SUMMARY, TECHNICAL SKILLS, EXPERIENCE, PROJECTS, EDUCATION).

    Resume:
    {resume_text[:4500]}

    Job Description:
    {job_description[:3000]}
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Optimization error: {str(e)}"

def create_resume_pdf(optimized_resume):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    style_sheet = getSampleStyleSheet()
    
    heading_style = ParagraphStyle(
        'Heading2_Custom',
        parent=style_sheet['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=10,
        spaceAfter=4
    )
    normal_style = ParagraphStyle(
        'Body_Custom',
        parent=style_sheet['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=3
    )

    content = []
    cleaned_text = re.sub(r"\*\*(.*?)\*\*", r"\1", optimized_resume)
    cleaned_text = re.sub(r"#+\s*", "", cleaned_text)

    for line in cleaned_text.split("\n"):
        line_str = line.strip()
        if not line_str:
            continue
        if line_str.isupper() and len(line_str) < 60:
            content.append(Paragraph(line_str, heading_style))
            content.append(Spacer(1, 2))
        else:
            safe_line = line_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            content.append(Paragraph(safe_line, normal_style))
            content.append(Spacer(1, 2))

    doc.build(content)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

# ---------------------------------------------------------
# Input Section (Two Columns)
# ---------------------------------------------------------
col_input1, col_input2 = st.columns([1, 1], gap="large")

with col_input1:
    st.markdown("### 📄 1. Upload Resume")
    uploaded_resume = st.file_uploader("Select Resume (PDF Format)", type=["pdf"])
    
    # Preset sample toggle
    use_sample = st.checkbox("⚡ Use Sample Software Engineer Resume & JD", value=st.session_state.sample_mode)

with col_input2:
    st.markdown("### 📋 2. Target Job Description")
    default_jd = SAMPLE_JOB_DESCRIPTION if use_sample else ""
    job_description = st.text_area(
        "Paste Job Description Here",
        value=default_jd,
        height=180,
        placeholder="Paste the target job description or role requirements here..."
    )

st.markdown("<br>", unsafe_allow_html=True)
analyze_button = st.button("🚀 Analyze Resume & Job Alignment", use_container_width=True)

# ---------------------------------------------------------
# Analysis Trigger Logic
# ---------------------------------------------------------
if analyze_button:
    if not use_sample and uploaded_resume is None:
        st.warning("⚠️ Please upload a PDF resume or check 'Use Sample Software Engineer Resume & JD'.")
    elif not job_description.strip():
        st.warning("⚠️ Please enter or paste a job description.")
    else:
        with st.spinner("🔍 Processing resume text and running NLP keyword alignment..."):
            if use_sample:
                resume_text = SAMPLE_RESUME_TEXT
            else:
                resume_text = extract_resume_text(uploaded_resume)

            # Calculations
            text_score = calculate_match_score(resume_text, job_description)
            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_description)
            skill_score = calculate_skill_score(resume_skills, job_skills)
            
            # Weighted Final Match Score: 30% text similarity + 70% skill match ratio
            match_score = round((0.3 * text_score) + (0.7 * skill_score), 2)
            
            matched_skills = list(set(resume_skills) & set(job_skills))
            missing_skills = list(set(job_skills) - set(resume_skills))
            
            # LLM Insights
            suggested_role = suggest_role(resume_text)
            feedback = generate_feedback(resume_text, job_description, match_score, missing_skills, suggested_role)

            # Persist to Session State
            st.session_state.analysis_done = True
            st.session_state.resume_text = resume_text
            st.session_state.job_description = job_description
            st.session_state.match_score = match_score
            st.session_state.text_score = text_score
            st.session_state.skill_score = skill_score
            st.session_state.matched_skills = matched_skills
            st.session_state.missing_skills = missing_skills
            st.session_state.feedback = feedback
            st.session_state.suggested_role = suggested_role
            st.session_state.optimized_resume = ""

# ---------------------------------------------------------
# Results Dashboard Rendering
# ---------------------------------------------------------
if st.session_state.analysis_done:
    st.markdown("<hr style='margin: 32px 0; border: none; border-top: 1px solid #cbd5e1;'>", unsafe_allow_html=True)
    st.markdown("## 📊 Analysis & Talent Intelligence Dashboard")

    # Read values from session state
    match_score = st.session_state.match_score
    text_score = st.session_state.text_score
    skill_score = st.session_state.skill_score
    matched_skills = st.session_state.matched_skills
    missing_skills = st.session_state.missing_skills
    feedback = st.session_state.feedback
    suggested_role = st.session_state.suggested_role

    # Create Dashboard Tabs
    tab_overview, tab_skills, tab_feedback, tab_optimize = st.tabs([
        "📊 Match Overview",
        "📌 Skill Gap Analysis",
        "💬 AI Recruiter Insights",
        "📝 ATS Resume Generator"
    ])

    # ---------------- TAB 1: OVERVIEW ----------------
    with tab_overview:
        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            score_color = "emerald" if match_score >= 70 else ("amber" if match_score >= 40 else "rose")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Overall Match Score</div>
                <div class="metric-value {score_color}">{match_score}%</div>
                <p style="font-size: 0.8rem; color: #64748b; margin-top: 8px;">Weighted combination of content similarity and keyword match</p>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Content Similarity</div>
                <div class="metric-value indigo">{text_score}%</div>
                <p style="font-size: 0.8rem; color: #64748b; margin-top: 8px;">TF-IDF Cosine Similarity Vector Alignment</p>
            </div>
            """, unsafe_allow_html=True)

        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Skill Coverage</div>
                <div class="metric-value emerald">{skill_score}%</div>
                <p style="font-size: 0.8rem; color: #64748b; margin-top: 8px;">{len(matched_skills)} of {len(matched_skills) + len(missing_skills)} required skills matched</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Match Progress Bar")
        st.progress(match_score / 100.0)

    # ---------------- TAB 2: SKILLS GAP ----------------
    with tab_skills:
        col_sk1, col_sk2 = st.columns(2, gap="large")

        with col_sk1:
            st.markdown(f"### ✅ Matched Skills ({len(matched_skills)})")
            if matched_skills:
                chips_html = "".join([f'<span class="skill-chip matched">✓ {s.title()}</span>' for s in matched_skills])
                st.markdown(f'<div class="skills-container">{chips_html}</div>', unsafe_allow_html=True)
            else:
                st.info("No matching skills detected from database taxonomy.")

        with col_sk2:
            st.markdown(f"### ❌ Missing Skills ({len(missing_skills)})")
            if missing_skills:
                chips_html = "".join([f'<span class="skill-chip missing">✕ {s.title()}</span>' for s in missing_skills])
                st.markdown(f'<div class="skills-container">{chips_html}</div>', unsafe_allow_html=True)
            else:
                st.success("🎉 No missing skills detected! Great alignment.")

    # ---------------- TAB 3: AI FEEDBACK & ROLE ----------------
    with tab_feedback:
        col_fb1, col_fb2 = st.columns([1.2, 0.8], gap="large")

        with col_fb1:
            st.markdown("### 💬 Executive Recruiter Feedback")
            st.markdown(f"""
            <div class="feedback-box">
                {feedback}
            </div>
            """, unsafe_allow_html=True)

        with col_fb2:
            st.markdown("### 🎯 Recommended Career Trajectory")
            st.markdown(f"""
            <div class="role-card">
                <div style="font-size: 0.8rem; font-weight: 700; color: #9333ea; text-transform: uppercase;">Best Suited Role</div>
                <div class="role-title">{suggested_role.split('Why:')[0].replace('Career Role:', '').strip()}</div>
                <p style="font-size: 0.9rem; color: #4b5563; margin-top: 8px;">
                    {suggested_role.split('Why:')[1].strip() if 'Why:' in suggested_role else ''}
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ---------------- TAB 4: ATS OPTIMIZATION & PDF ----------------
    with tab_optimize:
        st.markdown("### ✨ ATS-Optimized Resume Rewriter")
        st.write("Generate an upgraded version of your resume tailored to the job description while preserving 100% truthfulness.")

        if st.button("✨ Generate Optimized Resume", key="gen_opt_btn"):
            with st.spinner("Generating ATS-optimized resume using Groq LLM..."):
                optimized_text = generate_optimized_resume(
                    st.session_state.resume_text,
                    st.session_state.job_description
                )
                st.session_state.optimized_resume = optimized_text

        if st.session_state.optimized_resume:
            st.success("✅ Optimized resume has been generated! Click below to download your ATS-tailored PDF resume.")

            # Generate PDF
            pdf_bytes = create_resume_pdf(st.session_state.optimized_resume)

            st.download_button(
                label="📥 Download Optimized Resume PDF",
                data=pdf_bytes,
                file_name="ATS_Optimized_Resume.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.info("Click **'✨ Generate Optimized Resume'** to rewrite your resume with high ATS alignment.")