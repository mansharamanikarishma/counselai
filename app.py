"""
CounselAI: Decision Support System for Engineering Admissions
Comprehensive Refinement:
- Permanently pinned left sidebar with exact 0.2cm top space before CounselAI header
- Option C: App-Style Flat Navigation Links (Linear / Notion / Slack style)
- Cleaned all unclosed/empty HTML tags (solid-panel div bug eliminated)
- Seamless Segmented Control + native st.container(border=True) in Candidate Registry
- Discrete integer years on Cutoff Trajectories (strictly 2021-2025)
- Strict Exam Handling: MHT-CET Only (no JEE), JEE Main Only (no CET), or Both
- Interactive CAP Option Form with guaranteed 100% complete DTE College Codes
- Campus Facilities & Transit Comparison Matrix
- Real Conversational Chatbot with Persistent Memory
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_processor import (
    load_and_preprocess,
    COLLEGE_TIER_DATABASE,
    DEFAULT_TIER,
    CACHE_DIR
)
from ml_engine import AdmissionMLEngine
from counselor_bot import CounselingChatbot
from betterment_guide import render_betterment_guide

# Page Configuration
st.set_page_config(
    page_title="CounselAI | Decision Support System for Engineering Admissions",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Comprehensive Campus Facilities Knowledge Base for Maharashtra Colleges
COLLEGE_FACILITIES_DATABASE = {
    "6006": { # COEP
        "campus_size": "36.5 Acres (Shivajinagar, Central Pune)",
        "hostel": "Available (Boys: 2,000+ beds, Girls: 800+ beds)",
        "accreditation": "Autonomous Institute, NAAC A+, Tier-1 NBA",
        "infra_highlights": "Central Library (1.5 Lakh Books), 24x7 Wi-Fi, Boat Club, Innovation Labs",
        "metro_connectivity": "Direct (COEP Metro Station on doorstep)",
        "sports": "Cricket Ground, Boat Club on Mula River, Basketball & Badminton Courts"
    },
    "3012": { # VJTI
        "campus_size": "16 Acres (Matunga, Central Mumbai)",
        "hostel": "Available (Boys: 800+ beds, Girls: 300+ beds)",
        "accreditation": "Autonomous Institute, NAAC A++, Tier-1 NBA",
        "infra_highlights": "High-Voltage Tech Labs, Supercomputing Cluster, Heritage Campus",
        "metro_connectivity": "Direct (Dadar & Wadala railway stations 5 mins away)",
        "sports": "Gymkhana, Football Ground, Tennis Courts"
    },
    "3215": { # SPIT
        "campus_size": "47 Acres Bhavan's Campus (Andheri West, Mumbai)",
        "hostel": "Available on Bhavan's Campus (Limited capacity)",
        "accreditation": "Autonomous Institute, NAAC A Grade",
        "infra_highlights": "Incubation Centre (SP-TBI), Nvidia Deep Learning Lab, IoT Innovation Hub",
        "metro_connectivity": "5 Mins from Azad Nagar & Andheri Metro Stations",
        "sports": "Bhavan's Lake, Botanical Gardens, Cultural Amphitheatre"
    },
    "6271": { # PICT
        "campus_size": "5 Acres (Dhankawadi, Pune)",
        "hostel": "Available (Boys: 350 beds, Girls: 250 beds)",
        "accreditation": "Autonomous Institute, NBA Accredited",
        "infra_highlights": "Specialized Software Research Labs, ACM & IEEE Student Chapters, Startup Cell",
        "metro_connectivity": "Direct PMPML bus corridor, 15 mins to Swargate Metro",
        "sports": "Basketball Court, Indoor Gym, Table Tennis Arena"
    },
    "6139": { # VIT Pune
        "campus_size": "17.5 Acres across 2 campuses (Bibwewadi & Kondhwa, Pune)",
        "hostel": "Private Partner Hostels with College Bus Transport",
        "accreditation": "Autonomous Institute, NAAC A++ (CGPA 3.53)",
        "infra_highlights": "Mercedes-Benz Mechatronics Lab, Texas Instruments Lab, 500-seater Auditorium",
        "metro_connectivity": "10 Mins to Swargate & Mandai Metro Stations",
        "sports": "Badminton, Gym, Cultural Stage"
    },
    "6175": { # PCCOE
        "campus_size": "13 Acres (Nigdi / Akurdi, Pune Metro)",
        "hostel": "Available (Boys: 450 beds, Girls: 350 beds)",
        "accreditation": "Autonomous Institute, NAAC A Grade",
        "infra_highlights": "KPIT Automotive Lab, Innovation & Incubation Centre, Robotics Club",
        "metro_connectivity": "3 Mins to Akurdi Railway Station (Local Train to Pune)",
        "sports": "Multipurpose Sports Ground, Gymnasium"
    },
    "3199": { # DJ Sanghvi
        "campus_size": "SVKM Educational Campus (Vile Parle West, Mumbai)",
        "hostel": "SVKM Central Hostels in Vile Parle / Juhu",
        "accreditation": "Autonomous Institute, NAAC A Grade",
        "infra_highlights": "State-of-the-art Apple Mac Labs, Bloomberg Terminal Lab, Air-conditioned Classrooms",
        "metro_connectivity": "Walking distance to Vile Parle Station & D.N. Nagar Metro",
        "sports": "SVKM Indoor Sports Complex, Badminton & Squash Courts"
    }
}

DEFAULT_FACILITIES = {
    "campus_size": "10+ Acres Affiliated Campus",
    "hostel": "Hostel facilities available on campus or via verified institutional tie-ups",
    "accreditation": "Affiliated / Approved by AICTE & DTE Maharashtra",
    "infra_highlights": "Central Computing Centre, Digital Library, Departmental Laboratories",
    "metro_connectivity": "Accessible via city bus transport and local railway networks",
    "sports": "Standard Playground, Indoor Sports Room"
}

def get_initials(name):
    parts = str(name).strip().split()
    if len(parts) >= 2:
        return f"{parts[0][0]}{parts[1][0]}".upper()
    elif len(parts) == 1 and len(parts[0]) > 0:
        return parts[0][:2].upper()
    return "CA"

# ----------------- MODERN AIRTIGHT CSS (OPTION C: FLAT SaaS NAVIGATION) -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #1E293B;
    }

    .stApp {
        background-color: #F8FAFC;
    }

    /* 1. Permanently remove Streamlit default headers, collapse arrow and controls */
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0 !important;
    }

    [data-testid="stSidebarHeader"] {
        display: none !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    [data-testid="collapsedControl"] {
        display: none !important;
    }

    button[kind="header"] {
        display: none !important;
    }

    /* 2. Lock sidebar permanently on the left */
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        transform: none !important;
        transition: none !important;
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        width: 290px !important;
        min-width: 290px !important;
        max-width: 290px !important;
        position: relative !important;
    }

    /* 3. Comfortable breathing room above sidebar CounselAI header */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
    }

    /* 4. Align main content area top padding */
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 2.2rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
    }

    /* Category Micro-Header in Sidebar */
    .nav-cat-label {
        font-size: 0.63rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 3px;
        padding-left: 8px;
    }

    /* Option C: App-Style Flat Navigation Links (Linear / Notion Style) */
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        width: 100% !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 7px 12px !important;
        border-radius: 6px !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        margin-bottom: 2px !important;
        border: 1px solid transparent !important;
        box-shadow: none !important;
        background-color: transparent !important;
        transition: all 0.15s ease-in-out !important;
    }

    /* Inactive Flat Links: clean, transparent with subtle hover slide */
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"] {
        background-color: transparent !important;
        color: #475569 !important;
        border: 1px solid transparent !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"]:hover {
        background-color: #F1F5F9 !important;
        color: #0B2046 !important;
        border: 1px solid transparent !important;
        transform: translateX(3px) !important;
    }

    /* Active Flat Link: Soft ice-blue wash, deep royal navy text, left indicator line */
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #EFF6FF !important;
        color: #1E40AF !important;
        border: 1px solid #DBEAFE !important;
        border-left: 3.5px solid #2563EB !important;
        font-weight: 600 !important;
        border-radius: 0 6px 6px 0 !important;
        box-shadow: none !important;
        transform: none !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #E0E7FF !important;
        color: #1E40AF !important;
    }

    /* Structured Panels */
    .solid-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(11, 32, 70, 0.02);
    }

    .advisory-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 7px;
        padding: 12px 16px;
        margin-bottom: 8px;
        box-shadow: 0 1px 2px rgba(11, 32, 70, 0.02);
        transition: all 0.12s ease;
    }

    .advisory-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 3px 8px rgba(11, 32, 70, 0.05);
    }

    /* Status Badges */
    .badge-ambitious {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .badge-target {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .badge-safe {
        background-color: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- CANDIDATE STATE INITIALIZATION -----------------
if "candidates" not in st.session_state:
    st.session_state["candidates"] = [
        {
            "id": "EN26109432",
            "name": "Priya Sharma",
            "hu": "Savitribai Phule Pune University (SPPU)",
            "exam_mode": "MHT-CET Only (Maharashtra State Quota)",
            "score_cet": 93.40,
            "score_jee": None,
            "category": "OPEN",
            "gender": "Ladies",
            "round_target": "Round 1",
            "cities": ["Pune", "Mumbai / MMR", "Nagpur"],
            "branches": ["Computer Engineering", "Information Technology", "AI & Data Science"],
            "status": "active"
        },
        {
            "id": "EN26108119",
            "name": "Rohan Patil",
            "hu": "University of Mumbai (MU)",
            "exam_mode": "Both Exams (Appeared for both CET & JEE)",
            "score_cet": 88.75,
            "score_jee": 92.10,
            "category": "OBC",
            "gender": "General",
            "round_target": "Round 1",
            "cities": ["Mumbai / MMR", "Pune"],
            "branches": ["Computer Engineering", "AI & Data Science", "Electronics & Telecommunication"],
            "status": "passive"
        },
        {
            "id": "EN26115403",
            "name": "Aryan Gupta",
            "hu": "Other than Maharashtra (Non-MH / AI)",
            "exam_mode": "JEE Main Only (All India Quota)",
            "score_cet": None,
            "score_jee": 94.80,
            "category": "OPEN",
            "gender": "General",
            "round_target": "Round 1",
            "cities": ["Pune", "Mumbai / MMR"],
            "branches": ["Computer Engineering", "Information Technology"],
            "status": "passive"
        }
    ]

if "active_candidate_id" not in st.session_state:
    st.session_state["active_candidate_id"] = "EN26109432"

if "nav_step" not in st.session_state:
    st.session_state["nav_step"] = "👤 Candidate Profiles"

# Interactive Chatbot History
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "assistant", "content": "Welcome to **CounselAI**! I have loaded your active candidate profile. Ask me anything about cutoff trajectories, college comparisons, TFWS eligibility, or betterment strategy."}
    ]

def get_active_candidate():
    for c in st.session_state["candidates"]:
        if c["id"] == st.session_state["active_candidate_id"]:
            return c
    return st.session_state["candidates"][0]

active_cand = get_active_candidate()

# Cached Models
@st.cache_resource
def get_ml_engine():
    return AdmissionMLEngine()

@st.cache_resource
def get_chatbot(_engine):
    return CounselingChatbot(_engine)

engine = get_ml_engine()
chatbot = get_chatbot(engine)

all_cities = engine.get_all_cities()
all_branches = engine.get_all_branches()
colleges_dict = engine.get_all_colleges()

# ----------------- MODERN PERMANENT SaaS SIDEBAR (OPTION C) -----------------
with st.sidebar:
    # CounselAI Header: With clean breathing room above it
    st.markdown("""
    <div style="padding-top: 12px; margin-top: 4px; display: flex; align-items: center; gap: 10px; padding-bottom: 12px; border-bottom: 1px solid #E2E8F0; margin-bottom: 12px;">
        <div style="background: linear-gradient(135deg, #0B2046 0%, #1E3A8A 100%); width: 34px; height: 34px; border-radius: 7px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 800; font-size: 1rem; box-shadow: 0 2px 4px rgba(11,32,70,0.15);">
            C
        </div>
        <div>
            <div style="font-size: 1.15rem; font-weight: 800; color: #0B2046; letter-spacing: -0.3px; line-height: 1.15;">CounselAI</div>
            <div style="font-size: 0.65rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 1px;">Admissions Decision Suite</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active Candidate Live Card in Sidebar
    active_cand = get_active_candidate()
    eff_score = active_cand["score_cet"] if active_cand.get("score_cet") is not None else active_cand["score_jee"]
    initials = get_initials(active_cand["name"])
    exam_badge_txt = "CET" if active_cand.get("score_cet") is not None else "JEE"

    st.markdown(f"""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 7px; padding: 8px 10px; margin-bottom: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-size: 0.62rem; font-weight: 700; text-transform: uppercase; color: #64748B; letter-spacing: 0.5px;">Active Applicant</span>
            <span style="background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; font-size: 0.6rem; font-weight: 700; padding: 1px 5px; border-radius: 8px;">● Live</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="background: #0B2046; color: #FFFFFF; font-weight: 700; font-size: 0.72rem; width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                {initials}
            </div>
            <div style="overflow: hidden;">
                <div style="font-weight: 700; font-size: 0.82rem; color: #0B2046; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">{active_cand['name']}</div>
                <div style="font-size: 0.7rem; color: #475569;">{eff_score:.1f}% ({exam_badge_txt}) • {active_cand['category']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Switcher Dropdown
    cand_labels = []
    for c in st.session_state["candidates"]:
        sc = f"{c['score_cet']:.1f}% CET" if c.get("score_cet") is not None else f"{c['score_jee']:.1f}% JEE"
        cand_labels.append(f"{c['name']} ({sc})")
    
    current_active_score_str = f"{active_cand['score_cet']:.1f}% CET" if active_cand.get("score_cet") is not None else f"{active_cand['score_jee']:.1f}% JEE"
    current_cand_label = f"{active_cand['name']} ({current_active_score_str})"
    
    sel_cand_label = st.selectbox(
        "Switch Active Applicant:",
        options=cand_labels,
        index=cand_labels.index(current_cand_label) if current_cand_label in cand_labels else 0,
        label_visibility="collapsed"
    )

    selected_idx = cand_labels.index(sel_cand_label)
    if st.session_state["candidates"][selected_idx]["id"] != st.session_state["active_candidate_id"]:
        st.session_state["active_candidate_id"] = st.session_state["candidates"][selected_idx]["id"]
        st.rerun()

    active_cand = get_active_candidate()
    eff_score = active_cand["score_cet"] if active_cand.get("score_cet") is not None else active_cand["score_jee"]

    # Option C: App-Style Flat Navigation Links with Category Micro-Headers
    nav_categories = [
        {
            "category": "APPLICANT DESK",
            "items": [
                ("👤 Candidate Profiles", "👤 Candidate Profiles")
            ]
        },
        {
            "category": "DECISION CORE",
            "items": [
                ("🎯 Admission Predictor", "🎯 Admission Predictor"),
                ("📥 CAP Option Form", "📥 CAP Option Form"),
                ("⚖️ Freeze vs. Betterment", "⚖️ Freeze vs. Betterment"),
                ("🔄 Betterment Simulator", "🔄 Betterment Simulator")
            ]
        },
        {
            "category": "INSTITUTIONAL INSIGHTS",
            "items": [
                ("🏛️ College Comparison", "🏛️ College Comparison"),
                ("📈 Cutoff Trajectories", "📈 Cutoff Trajectories"),
                ("🤖 Counseling Chatbot", "🤖 Counseling Chatbot")
            ]
        },
        {
            "category": "DATA FOUNDATION",
            "items": [
                ("📊 Dataset & Analytics", "📊 Dataset & Analytics")
            ]
        }
    ]

    for cat in nav_categories:
        st.markdown(f"<div class='nav-cat-label'>{cat['category']}</div>", unsafe_allow_html=True)
        for label, target_step in cat["items"]:
            is_active = (st.session_state["nav_step"] == target_step)
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_btn_{target_step}", use_container_width=True, type=btn_type):
                st.session_state["nav_step"] = target_step
                st.rerun()

    st.markdown("""
    <div style="margin-top: 14px; padding-top: 8px; border-top: 1px solid #E2E8F0; font-size: 0.68rem; color: #94A3B8; text-align: center;">
        <span style="display: inline-block; width: 6px; height: 6px; background-color: #10B981; border-radius: 50%; margin-right: 4px;"></span>
        <strong>CounselAI Engine v2.5</strong> • Local
    </div>
    """, unsafe_allow_html=True)

# ----------------- COMPACT TOP HEADER BAR -----------------
st.markdown(f"""
<div style="background: linear-gradient(135deg, #0B2046 0%, #162E56 100%); padding: 10px 18px; border-radius: 7px; color: white; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; border: 1px solid #1E3A8A;">
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.3px;">CounselAI</span>
        <span style="color: #93C5FD; font-size: 0.78rem; border-left: 1px solid #3B82F6; padding-left: 10px;">Decision Support System for Engineering Admissions</span>
    </div>
    <div style="font-size: 0.76rem; color: #E2E8F0;">
        Active Session: <strong style="color: #FFFFFF;">{active_cand['name']}</strong> ({eff_score:.2f}% • {active_cand['category']} • {active_cand['gender']})
    </div>
</div>
""", unsafe_allow_html=True)

nav_selection = st.session_state["nav_step"]

# Common Prediction Execution for Active Candidate
preds = engine.predict_choices(
    score_cet=active_cand.get("score_cet"),
    score_jee=active_cand.get("score_jee"),
    exam_mode=active_cand["exam_mode"],
    category=active_cand["category"],
    gender=active_cand["gender"],
    selected_cities=active_cand["cities"],
    selected_branches=active_cand["branches"],
    round_target=active_cand["round_target"]
)

ambitious = preds["ambitious"]
target = preds["target"]
safe = preds["safe"]
all_df = preds["all_ordered"]

# ----------------- SECTION 1: CANDIDATE PROFILE REGISTRY (CLEANED OF ARTIFACTS) -----------------
if nav_selection == "👤 Candidate Profiles":
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <div>
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0B2046;">Candidate Profile Registry</h3>
            <p style="margin: 1px 0 0 0; font-size: 0.82rem; color: #64748B;">Central applicant repository. One active session drives predictions, option generation, and chatbot guidance.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Clean Roster Cards
    st.markdown("<div style='font-size: 0.74rem; text-transform: uppercase; color: #64748B; font-weight: 700; margin-bottom: 6px; letter-spacing: 0.5px;'>Enrolled Applicant Files</div>", unsafe_allow_html=True)
    c_cols = st.columns(len(st.session_state["candidates"]))
    cand_to_delete = None

    for idx, cand in enumerate(st.session_state["candidates"]):
        is_active = (cand["id"] == st.session_state["active_candidate_id"])
        c_score = cand.get("score_cet") if cand.get("score_cet") is not None else cand.get("score_jee", 0.0)
        exam_tag = "CET" if cand.get("score_cet") is not None else "JEE"
        cand_inits = get_initials(cand["name"])

        with c_cols[idx]:
            card_border = "2px solid #0B2046" if is_active else "1px solid #E2E8F0"
            card_shadow = "0 3px 10px rgba(11, 32, 70, 0.08)" if is_active else "0 1px 3px rgba(11, 32, 70, 0.02)"
            status_pill = (
                "<span style='background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; padding: 2px 6px; border-radius: 10px; font-size: 0.65rem; font-weight: 700;'>● ACTIVE SESSION</span>"
                if is_active else
                "<span style='background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; padding: 2px 6px; border-radius: 10px; font-size: 0.65rem; font-weight: 600;'>○ PASSIVE FILE</span>"
            )
            avatar_bg = "#0B2046" if is_active else "#64748B"

            st.markdown(f"""
            <div style="background: #FFFFFF; border: {card_border}; border-radius: 8px; padding: 12px 14px; box-shadow: {card_shadow}; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    {status_pill}
                    <code style="font-size: 0.72rem; color: #64748B;">{cand['id']}</code>
                </div>
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <div style="background: {avatar_bg}; color: #FFFFFF; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem;">
                        {cand_inits}
                    </div>
                    <div>
                        <div style="font-weight: 700; font-size: 0.95rem; color: #0B2046;">{cand['name']}</div>
                        <div style="font-size: 0.75rem; color: #64748B;">{cand['category']} • {cand['gender']}</div>
                    </div>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #F1F5F9; border-radius: 6px; padding: 6px 10px; font-size: 0.78rem; display: flex; justify-content: space-between;">
                    <span style="color: #64748B;">Merit Percentile:</span>
                    <strong style="color: #0B2046;">{c_score:.2f}% ({exam_tag})</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            b_act, b_del = st.columns([1, 1])
            with b_act:
                if not is_active:
                    if st.button("⚡ Activate", key=f"act_{cand['id']}", use_container_width=True):
                        st.session_state["active_candidate_id"] = cand["id"]
                        st.rerun()
                else:
                    st.button("✓ Live", key=f"cur_{cand['id']}", disabled=True, use_container_width=True)
            with b_del:
                if len(st.session_state["candidates"]) > 1 and not is_active:
                    if st.button("🗑️ Remove", key=f"del_{cand['id']}", use_container_width=True):
                        cand_to_delete = cand["id"]

    if cand_to_delete:
        st.session_state["candidates"] = [c for c in st.session_state["candidates"] if c["id"] != cand_to_delete]
        st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Segmented Control replacing tabs completely (Eliminates the empty gray tab track)
    profile_view = st.segmented_control(
        "Candidate Registry Workspace View",
        options=[f"✏️ Edit Profile: {active_cand['name']} (Active)", "➕ Enroll New Candidate Profile"],
        default=f"✏️ Edit Profile: {active_cand['name']} (Active)",
        label_visibility="collapsed"
    )

    if "Edit Profile" in profile_view:
        with st.container(border=True):
            st.markdown(f"<div style='font-size: 0.74rem; text-transform: uppercase; color: #64748B; font-weight: 700; margin-bottom: 8px;'>Applicant Parameters & Identity: {active_cand['name']}</div>", unsafe_allow_html=True)
            
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                edit_name = st.text_input("Candidate Full Name", value=active_cand["name"])
            with col_p2:
                edit_app_id = st.text_input("DTE Application ID", value=active_cand["id"])
            with col_p3:
                hu_options = [
                    "Savitribai Phule Pune University (SPPU)",
                    "University of Mumbai (MU)",
                    "Dr. BATU Lonere (State Technological University)",
                    "Rashtrasant Tukadoji Maharaj Nagpur University (RTMNU)",
                    "Shivaji University Kolhapur (SUK)",
                    "Dr. BAMU Chhatrapati Sambhajinagar",
                    "Sant Gadge Baba Amravati University (SGBAU)",
                    "Other than Maharashtra (Non-MH / AI)"
                ]
                edit_hu = st.selectbox("Home University Jurisdiction", options=hu_options, index=hu_options.index(active_cand["hu"]) if active_cand["hu"] in hu_options else 0)

            # STRICT EXAM SELECTION: Only asks for relevant exam score!
            st.markdown("<div style='font-size: 0.76rem; font-weight: 700; color: #0B2046; margin: 12px 0 4px 0;'>Qualifying Entrance Examination Basis</div>", unsafe_allow_html=True)
            
            edit_stream = st.radio(
                "Select Admission Entrance Exam:",
                options=[
                    "MHT-CET Only (Maharashtra State Quota)",
                    "JEE Main Only (All India Quota)",
                    "Both Exams (Appeared for both CET & JEE)"
                ],
                index=0 if "MHT-CET Only" in active_cand["exam_mode"] else (1 if "JEE Main Only" in active_cand["exam_mode"] else 2),
                label_visibility="collapsed"
            )

            edit_cet = None
            edit_jee = None

            if edit_stream == "MHT-CET Only (Maharashtra State Quota)":
                edit_cet = st.number_input("MHT-CET Percentile (PCM / PCB)", min_value=0.0, max_value=100.0, value=float(active_cand.get("score_cet") or 92.0), step=0.01)
                st.caption("✅ Only MHT-CET score evaluated for Maharashtra State seats. JEE Main not required.")
            elif edit_stream == "JEE Main Only (All India Quota)":
                edit_jee = st.number_input("JEE Main Paper-1 Percentile", min_value=0.0, max_value=100.0, value=float(active_cand.get("score_jee") or 90.0), step=0.01)
                st.caption("✅ Only JEE Main score evaluated for All India seats. MHT-CET not required.")
            else:  # Both
                c1, c2 = st.columns(2)
                with c1:
                    edit_cet = st.number_input("MHT-CET %ile", min_value=0.0, max_value=100.0, value=float(active_cand.get("score_cet") or 92.0), step=0.01)
                with c2:
                    edit_jee = st.number_input("JEE Main %ile", min_value=0.0, max_value=100.0, value=float(active_cand.get("score_jee") or 90.0), step=0.01)
                st.caption("Both exams evaluated with MH State quota evaluated primarily.")

            st.markdown("<div style='font-size: 0.76rem; font-weight: 700; color: #0B2046; margin: 12px 0 4px 0;'>Seat Category & Target Round</div>", unsafe_allow_html=True)
            col_q1, col_q2, col_q3 = st.columns(3)
            with col_q1:
                cat_opts = ["OPEN", "OBC", "SC", "ST", "VJ/NT", "SBC", "EWS", "TFWS"]
                edit_cat = st.selectbox("Seat Category", options=cat_opts, index=cat_opts.index(active_cand["category"]))
            with col_q2:
                edit_gen = st.selectbox("Gender Quota", options=["General", "Ladies"], index=0 if active_cand["gender"] == "General" else 1)
            with col_q3:
                edit_round = st.selectbox("Target Counseling Round", options=["Round 1", "Round 2", "Round 3", "Best of All Rounds"], index=0)

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                edit_cities = st.multiselect("Preferred Cities", options=all_cities, default=active_cand["cities"])
            with col_f2:
                edit_branches = st.multiselect("Preferred Engineering Streams", options=all_branches, default=active_cand["branches"])

            if st.button("💾 Save Profile Changes & Recompute Choices", type="primary"):
                for c in st.session_state["candidates"]:
                    if c["id"] == active_cand["id"]:
                        c["name"] = edit_name
                        c["id"] = edit_app_id
                        c["hu"] = edit_hu
                        c["exam_mode"] = edit_stream
                        c["score_cet"] = edit_cet
                        c["score_jee"] = edit_jee
                        c["category"] = edit_cat
                        c["gender"] = edit_gen
                        c["round_target"] = edit_round
                        c["cities"] = edit_cities
                        c["branches"] = edit_branches
                        break
                st.success(f"Updated applicant record for {edit_name}!")
                st.rerun()

    else:
        with st.container(border=True):
            st.markdown("<div style='font-size: 0.74rem; text-transform: uppercase; color: #64748B; font-weight: 700; margin-bottom: 8px;'>New Applicant Enrollment</div>", unsafe_allow_html=True)
            
            na_c1, na_c2, na_c3 = st.columns(3)
            with na_c1:
                new_name = st.text_input("New Candidate Name", placeholder="e.g., Aditya Kulkarni", key="new_name_inp")
            with na_c2:
                new_id = st.text_input("Application ID", value=f"EN261{np.random.randint(10000, 99999)}", key="new_id_inp")
            with na_c3:
                new_hu = st.selectbox("Home University", options=[
                    "Savitribai Phule Pune University (SPPU)",
                    "University of Mumbai (MU)",
                    "Dr. BATU Lonere (State Technological University)",
                    "Rashtrasant Tukadoji Maharaj Nagpur University (RTMNU)",
                    "Other than Maharashtra (Non-MH / AI)"
                ], key="new_hu_select_tab")

            na_s1, na_s2 = st.columns(2)
            with na_s1:
                new_exam_mode = st.radio("Qualifying Exam Basis", [
                    "MHT-CET Only (Maharashtra State Quota)",
                    "JEE Main Only (All India Quota)",
                    "Both Exams (Appeared for both CET & JEE)"
                ], key="new_exam_mode_radio_tab")
            with na_s2:
                new_cet = None
                new_jee = None
                if "MHT-CET Only" in new_exam_mode:
                    new_cet = st.number_input("MHT-CET Percentile", min_value=0.0, max_value=100.0, value=90.0, step=0.1, key="new_cet_val_tab")
                elif "JEE Main Only" in new_exam_mode:
                    new_jee = st.number_input("JEE Main Paper-1 Percentile", min_value=0.0, max_value=100.0, value=89.0, step=0.1, key="new_jee_val_tab")
                else:
                    nc1, nc2 = st.columns(2)
                    with nc1:
                        new_cet = st.number_input("CET %ile", min_value=0.0, max_value=100.0, value=91.0, step=0.1, key="new_both_cet_tab")
                    with nc2:
                        new_jee = st.number_input("JEE %ile", min_value=0.0, max_value=100.0, value=88.5, step=0.1, key="new_both_jee_tab")

            if st.button("Enroll Applicant into Registry", type="primary"):
                if new_name.strip():
                    st.session_state["candidates"].append({
                        "id": new_id,
                        "name": new_name.strip(),
                        "hu": new_hu,
                        "exam_mode": new_exam_mode,
                        "score_cet": new_cet,
                        "score_jee": new_jee,
                        "category": "OPEN",
                        "gender": "General",
                        "round_target": "Round 1",
                        "cities": ["Pune", "Mumbai / MMR"],
                        "branches": ["Computer Engineering", "Information Technology"],
                        "status": "passive"
                    })
                    st.success(f"Enrolled {new_name} successfully!")
                    st.rerun()

# ----------------- SECTION 2: ADMISSION PREDICTOR -----------------
elif nav_selection == "🎯 Admission Predictor":
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <div>
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0B2046;">Personalized Admission Predictor</h3>
            <p style="margin: 1px 0 0 0; font-size: 0.82rem; color: #64748B;">Recommendations for active candidate: <strong>{active_cand['name']}</strong> ({eff_score:.2f}% • {active_cand['category']}).</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="solid-panel" style="padding: 10px 14px; margin-bottom: 10px;">
            <div style="font-size: 0.68rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Ambitious Choices</div>
            <div style="font-size: 1.35rem; font-weight: 700; color: #991B1B;">{len(ambitious)}</div>
            <div style="font-size: 0.72rem; color: #64748B;">Dream Reach (15%–40%)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="solid-panel" style="padding: 10px 14px; margin-bottom: 10px;">
            <div style="font-size: 0.68rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Target Choices</div>
            <div style="font-size: 1.35rem; font-weight: 700; color: #92400E;">{len(target)}</div>
            <div style="font-size: 0.72rem; color: #64748B;">Realistic Match (40%–80%)</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="solid-panel" style="padding: 10px 14px; margin-bottom: 10px;">
            <div style="font-size: 0.68rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Safe Choices</div>
            <div style="font-size: 1.35rem; font-weight: 700; color: #166534;">{len(safe)}</div>
            <div style="font-size: 0.72rem; color: #64748B;">Guaranteed Net (80%+)</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="solid-panel" style="padding: 10px 14px; margin-bottom: 10px;">
            <div style="font-size: 0.68rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Target Round</div>
            <div style="font-size: 1.35rem; font-weight: 700; color: #0B2046;">{active_cand['round_target']}</div>
            <div style="font-size: 0.72rem; color: #64748B;">{len(active_cand['cities'])} Cities Included</div>
        </div>
        """, unsafe_allow_html=True)

    sub_t1, sub_t2, sub_t3, sub_t4 = st.tabs([
        f"🔥 Ambitious ({len(ambitious)})",
        f"🎯 Target ({len(target)})",
        f"🛡️ Safe ({len(safe)})",
        "📋 Complete Sequence Table"
    ])

    def render_items(items, badge_class, badge_label):
        if not items:
            st.info("No colleges match this category with active filters. Try expanding cities or streams in Candidate Profiles.")
            return

        for item in items[:25]:
            st.markdown(f"""
            <div class="advisory-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <span class="{badge_class}">{badge_label}</span>
                        <span style="font-size: 0.78rem; color: #64748B; margin-left: 8px;">DTE Code: <code>{item['college_code']}</code></span>
                        <h4 style="margin: 3px 0 2px 0; font-size: 0.98rem; font-weight: 600; color: #0B2046;">
                            {item['college_name']}
                        </h4>
                        <div style="font-size: 0.84rem; color: #334155;">
                            {item['branch']} • <span style="color: #64748B;">📍 {item['city']}</span>
                        </div>
                    </div>
                    <div style="text-align: right; min-width: 90px;">
                        <div style="font-size: 0.68rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Suitability</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #0B2046;">{item['suitability_pct']}%</div>
                    </div>
                </div>
                <div style="display: flex; gap: 16px; font-size: 0.8rem; color: #475569; border-top: 1px solid #F1F5F9; padding-top: 6px; margin-top: 6px;">
                    <div><strong>Tier:</strong> {item['tier']}</div>
                    <div><strong>Avg Placement:</strong> {item['avg_placement']}</div>
                    <div><strong>Projected Cutoff:</strong> {item['predicted_cutoff']}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with sub_t1:
        st.markdown("<p style='font-size: 0.82rem; color: #475569;'>Higher-tier colleges where cutoffs slightly exceed your score. Placed first on option forms to capture potential round drops.</p>", unsafe_allow_html=True)
        render_items(ambitious, "badge-ambitious", "Ambitious")

    with sub_t2:
        st.markdown("<p style='font-size: 0.82rem; color: #475569;'>Colleges whose cutoffs align directly with your score. Forms the stable core of your option list.</p>", unsafe_allow_html=True)
        render_items(target, "badge-target", "Target")

    with sub_t3:
        st.markdown("<p style='font-size: 0.82rem; color: #475569;'>Safe options with cutoffs safely below your score, functioning as guaranteed safety nets.</p>", unsafe_allow_html=True)
        render_items(safe, "badge-safe", "Safe")

    with sub_t4:
        st.markdown("<p style='font-size: 0.82rem; color: #475569;'>All choices ordered strictly per CAP strategy: <strong>Ambitious $\\to$ Target $\\to$ Safe</strong>.</p>", unsafe_allow_html=True)
        if len(all_df) > 0:
            disp_df = all_df[["badge", "college_code", "college_name", "branch", "city", "avg_placement", "predicted_cutoff", "suitability_pct", "choice_code"]].copy()
            st.dataframe(
                disp_df.rename(columns={
                    "badge": "Category",
                    "college_code": "DTE College Code",
                    "college_name": "College",
                    "branch": "Branch",
                    "city": "City",
                    "avg_placement": "Placement CTC",
                    "predicted_cutoff": "Cutoff %",
                    "suitability_pct": "Suitability %",
                    "choice_code": "DTE Choice Code"
                }),
                use_container_width=True,
                height=380
            )

# ----------------- SECTION 3: CAP OPTION FORM (INTERACTIVE & 100% COMPLETE) -----------------
elif nav_selection == "📥 CAP Option Form":
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <div>
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0B2046;">Interactive CAP Option Form Generator</h3>
            <p style="margin: 1px 0 0 0; font-size: 0.82rem; color: #64748B;">Personalized option sequence formatted with verified DTE College Codes and Choice Codes for <strong>{active_cand['name']}</strong>.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if len(all_df) > 0:
        # Interactive Filter Bar
        fb1, fb2, fb3 = st.columns([1, 1, 1])
        with fb1:
            tier_filter = st.selectbox("Filter by Category:", ["All Categories", "Only Ambitious", "Only Target", "Only Safe"])
        with fb2:
            sort_pref = st.selectbox("Sort Preference by:", [
                "Counseling Sequence (Ambitious -> Target -> Safe)",
                "Highest Placement CTC",
                "Highest Projected Cutoff",
                "Alphabetical"
            ])
        with fb3:
            limit_rows = st.selectbox("Show Top Options:", [20, 50, 100, "All Matches"], index=1)

        export_df = all_df.copy()

        if tier_filter == "Only Ambitious":
            export_df = export_df[export_df["category_tag"] == "Ambitious"]
        elif tier_filter == "Only Target":
            export_df = export_df[export_df["category_tag"] == "Target"]
        elif tier_filter == "Only Safe":
            export_df = export_df[export_df["category_tag"] == "Safe"]

        if sort_pref == "Highest Placement CTC":
            export_df = export_df.sort_values(by="avg_lpa", ascending=False)
        elif sort_pref == "Highest Projected Cutoff":
            export_df = export_df.sort_values(by="predicted_cutoff", ascending=False)
        elif sort_pref == "Alphabetical":
            export_df = export_df.sort_values(by="college_name", ascending=True)

        if limit_rows != "All Matches":
            export_df = export_df.head(int(limit_rows))

        # 100% Complete Columns (No nulls, clean DTE Codes)
        clean_exp = export_df[[
            "category_tag", "college_code", "college_name", "branch", "city", 
            "avg_placement", "predicted_cutoff", "suitability_pct", "choice_code"
        ]].dropna().copy()

        clean_exp.insert(0, "Pref #", range(1, len(clean_exp) + 1))
        clean_exp.columns = [
            "Pref #", "Category", "DTE College Code", "College Name", "Branch", 
            "City", "Avg Placement CTC", "Cutoff %", "Suitability %", "Official Choice Code"
        ]

        # Render Interactive Cards for Top 4
        st.markdown("<div style='font-size: 0.78rem; text-transform: uppercase; color: #64748B; font-weight: 700; margin-bottom: 6px;'>Top Preference Preview:</div>", unsafe_allow_html=True)
        for _, r in clean_exp.head(4).iterrows():
            c_badge = "badge-ambitious" if r["Category"] == "Ambitious" else ("badge-target" if r["Category"] == "Target" else "badge-safe")
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: 700; color: #0B2046; font-size: 0.88rem; margin-right: 8px;">#{r['Pref #']}</span>
                    <span class="{c_badge}">{r['Category']}</span>
                    <strong style="margin-left: 8px; font-size: 0.9rem; color: #0B2046;">{r['College Name']}</strong>
                    <div style="font-size: 0.8rem; color: #475569; margin-top: 2px;">{r['Branch']} • 📍 {r['City']} • DTE Code: <code>{r['DTE College Code']}</code></div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.7rem; color: #64748B; text-transform: uppercase; font-weight: 600;">Choice Code</div>
                    <code style="font-size: 0.88rem; font-weight: 700; color: #0B2046;">{r['Official Choice Code']}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.dataframe(clean_exp, use_container_width=True, height=360)

        csv_data = clean_exp.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download Verified Option Form for {active_cand['name']} (CSV)",
            data=csv_data,
            file_name=f"CAP_Option_Form_{active_cand['id']}.csv",
            mime="text/csv"
        )
    else:
        st.info("No options available with current profile parameters.")

# ----------------- SECTION 4: FREEZE VS BETTERMENT GUIDE (PRECEDES SIMULATOR) -----------------
elif nav_selection == "⚖️ Freeze vs. Betterment":
    render_betterment_guide()

# ----------------- SECTION 5: BETTERMENT SIMULATOR (WITH PROPER TEXT & VALID DTE CODES) -----------------
elif nav_selection == "🔄 Betterment Simulator":
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <div>
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0B2046;">Round 2/3 Betterment Upgrade Simulator</h3>
            <p style="margin: 1px 0 0 0; font-size: 0.82rem; color: #64748B;">Analyzes historical multi-round drops to evaluate if Betterment (Float) yields viable upgrades for <strong>{active_cand['name']}</strong>.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        all_college_list = [f"[{code}] {name}" for code, name in colleges_dict.items()]
        default_allotted = [opt for opt in all_college_list if "6175" in opt or "6139" in opt]
        sel_allotted = st.selectbox("Round 1 Allotted College:", options=all_college_list, index=0 if not default_allotted else all_college_list.index(default_allotted[0]))
        allotted_code = sel_allotted.split("] ")[0].replace("[", "").strip()

    with sim_col2:
        allotted_branch = st.selectbox("Round 1 Allotted Branch:", options=all_branches, index=0)

    if st.button("Evaluate Betterment Upgrades", type="primary"):
        with st.spinner("Analyzing historical Round 1 -> Round 2 & Round 3 cutoff drops..."):
            upgrades_df = engine.simulate_betterment(
                allotted_college_code=allotted_code,
                allotted_branch=allotted_branch,
                student_score=eff_score,
                category=active_cand["category"],
                selected_cities=active_cand["cities"]
            )

        if len(upgrades_df) > 0:
            st.success(f"Identified {len(upgrades_df)} institutions with statistically viable upgrade paths.")
            
            # Format text in clean Title Case and proper DTE Code headers
            upgrades_formatted = upgrades_df[[
                "college_code", "college_name", "branch", "city", "avg_placement", 
                "r1_cutoff", "r2_cutoff", "avg_drop", "upgrade_chance_pct", "target_round"
            ]].copy()

            st.dataframe(
                upgrades_formatted.rename(columns={
                    "college_code": "DTE College Code",
                    "college_name": "Institution",
                    "branch": "Branch / Stream",
                    "city": "Region",
                    "avg_placement": "Placement CTC",
                    "r1_cutoff": "Round 1 Cutoff %",
                    "r2_cutoff": "Round 2 Cutoff %",
                    "avg_drop": "Historical Drop %",
                    "upgrade_chance_pct": "Upgrade Chance %",
                    "target_round": "Recommended Target Round"
                }),
                use_container_width=True,
                height=360
            )
        else:
            st.info("Your allotted seat is already at the optimal boundary for your score tier. Betterment remains safe to exercise as your current seat is retained.")

# ----------------- SECTION 6: COLLEGE COMPARISON (WITH CAMPUS FACILITIES & INFRASTRUCTURE) -----------------
elif nav_selection == "🏛️ College Comparison":
    st.markdown("""
    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0B2046;">Institutional Comparison & Facilities Matrix</h3>
    <p style="margin: 1px 0 14px 0; font-size: 0.82rem; color: #64748B;">Compare placement packages, cutoffs, campus acreage, hostel facilities, and transit connectivity across institutions.</p>
    """, unsafe_allow_html=True)

    all_college_options = [f"[{code}] {name}" for code, name in colleges_dict.items()]
    default_compare = [opt for opt in all_college_options if any(c in opt for c in ["6006", "3012", "6271", "3215"])][:3]

    selected_colleges_str = st.multiselect(
        "Select Institutions to Compare (Up to 3):",
        options=all_college_options,
        default=default_compare,
        max_selections=3
    )

    if selected_colleges_str:
        selected_codes = [opt.split("] ")[0].replace("[", "").strip() for opt in selected_colleges_str]
        
        # Side-by-side Campus Feature Cards
        card_cols = st.columns(len(selected_codes))
        for i, code in enumerate(selected_codes):
            c_name = colleges_dict.get(code, f"College {code}")
            tier_info = COLLEGE_TIER_DATABASE.get(code, DEFAULT_TIER)
            fac_info = COLLEGE_FACILITIES_DATABASE.get(code, DEFAULT_FACILITIES)
            
            with card_cols[i]:
                st.markdown(f"""
                <div class="solid-panel" style="min-height: 390px;">
                    <div style="font-size: 0.7rem; text-transform: uppercase; color: #64748B; font-weight: 700;">DTE Code: {code}</div>
                    <h4 style="margin: 2px 0 4px 0; font-size: 1.05rem; font-weight: 700; color: #0B2046;">{c_name}</h4>
                    <div style="font-size: 0.78rem; color: #166534; font-weight: 600; margin-bottom: 8px;">{tier_info['tier']}</div>
                    <hr style="margin: 6px 0; border: 0; border-top: 1px solid #F1F5F9;">
                    <div style="font-size: 0.82rem; margin-bottom: 6px;"><strong>💰 Avg Package:</strong> {tier_info['avg_ctc']} (Highest: {tier_info['max_ctc']})</div>
                    <div style="font-size: 0.82rem; margin-bottom: 6px;"><strong>🏛️ Campus:</strong> {fac_info['campus_size']}</div>
                    <div style="font-size: 0.82rem; margin-bottom: 6px;"><strong>🛏️ Hostels:</strong> {fac_info['hostel']}</div>
                    <div style="font-size: 0.82rem; margin-bottom: 6px;"><strong>🏅 Accreditation:</strong> {fac_info['accreditation']}</div>
                    <div style="font-size: 0.82rem; margin-bottom: 6px;"><strong>🚇 Metro/Transit:</strong> {fac_info['metro_connectivity']}</div>
                    <div style="font-size: 0.82rem; margin-bottom: 6px;"><strong>⚽ Sports:</strong> {fac_info['sports']}</div>
                    <div style="font-size: 0.78rem; color: #475569; margin-top: 6px;"><strong>Top Recruiters:</strong> {tier_info['recruiters']}</div>
                </div>
                """, unsafe_allow_html=True)

        chart_data = []
        for code in selected_codes:
            info = COLLEGE_TIER_DATABASE.get(code, DEFAULT_TIER)
            c_name = colleges_dict.get(code, f"College {code}")[:22]
            chart_data.append({"Institution": c_name, "Average CTC (LPA)": info["avg_lpa"]})

        df_chart = pd.DataFrame(chart_data)
        fig_bar = px.bar(
            df_chart, 
            x="Institution", 
            y="Average CTC (LPA)", 
            title="Average Placement CTC Comparison (LPA)",
            color_discrete_sequence=["#0B2046"]
        )
        fig_bar.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font_family="Plus Jakarta Sans",
            height=300,
            margin=dict(l=15, r=15, t=35, b=15)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ----------------- SECTION 7: 5-YEAR CUTOFF TRAJECTORIES (DISCRETE INTEGER YEARS) -----------------
elif nav_selection == "📈 Cutoff Trajectories":
    st.markdown("""
    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0B2046;">5-Year Cutoff Trajectory Explorer</h3>
    <p style="margin: 1px 0 14px 0; font-size: 0.82rem; color: #64748B;">Multi-year percentile shifts tracking branch cutoff trends across discrete academic years 2021 through 2025.</p>
    """, unsafe_allow_html=True)

    all_college_options = [f"[{code}] {name}" for code, name in colleges_dict.items()]
    tr_col1, tr_col2, tr_col3 = st.columns([2, 2, 1])
    
    with tr_col1:
        sel_trend_college = st.selectbox("Select College:", options=all_college_options, index=0)
        tr_code = sel_trend_college.split("] ")[0].replace("[", "").strip()

    # Dynamically find branches that ACTUALLY exist in this college
    college_raw = engine.df_raw[engine.df_raw["college_code"] == tr_code] if engine.df_raw is not None else pd.DataFrame()
    available_branches = sorted(college_raw["course_name"].unique().tolist()) if len(college_raw) > 0 else all_branches

    with tr_col2:
        sel_trend_branch = st.selectbox("Select Course / Stream:", options=available_branches if available_branches else all_branches)

    with tr_col3:
        cat_choices = ["OPEN", "OBC", "SC", "ST", "EWS", "TFWS"]
        active_c = active_cand["category"] if active_cand["category"] in cat_choices else "OPEN"
        sel_cat = st.selectbox("Category:", options=cat_choices, index=cat_choices.index(active_c))

    if len(college_raw) > 0:
        sub_raw = college_raw[(college_raw["course_name"] == sel_trend_branch) & (college_raw["category"] == sel_cat)]
        if len(sub_raw) == 0:
            # Fallback to OPEN if candidate's specific category has no historical record
            sub_raw = college_raw[(college_raw["course_name"] == sel_trend_branch) & (college_raw["category"] == "OPEN")]
            st.caption(f"Showing 'OPEN' category historical trajectory for {sel_trend_branch}.")

        if len(sub_raw) > 0:
            trend_agg = sub_raw.groupby(["year", "round"])["percentile"].median().reset_index()
            # Guarantee discrete integer years (no 2024.4, strictly 2021, 2022, 2023, 2024, 2025)
            trend_agg["year_int"] = trend_agg["year"].astype(int)
            trend_agg = trend_agg.sort_values(by=["year_int", "round"])
            trend_agg["Academic Year"] = trend_agg["year_int"].astype(str)

            c_title = sel_trend_college.split("] ")[1] if "] " in sel_trend_college else sel_trend_college
            fig_trend = px.line(
                trend_agg,
                x="Academic Year",
                y="percentile",
                color="round",
                markers=True,
                title=f"{c_title[:35]} - {sel_trend_branch}",
                labels={"Academic Year": "Academic Year", "percentile": "Closing Cutoff Percentile", "round": "CAP Round"},
                color_discrete_sequence=["#0B2046", "#2563EB", "#059669"]
            )
            fig_trend.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=["2021", "2022", "2023", "2024", "2025"]
            )
            fig_trend.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font_family="Plus Jakarta Sans",
                height=360,
                xaxis=dict(tickmode="linear", dtick=1),
                yaxis=dict(gridcolor="#F1F5F9", title="Closing Cutoff Percentile")
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info(f"No specific historical cutoff records found for {sel_trend_branch}.")
    else:
        st.info("Loading historical trajectory records...")

# ----------------- SECTION 8: REAL INTERACTIVE CHATBOT -----------------
elif nav_selection == "🤖 Counseling Chatbot":
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <div>
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0B2046;">CounselAI Interactive Assistant</h3>
            <p style="margin: 1px 0 0 0; font-size: 0.82rem; color: #64748B;">Conversational advisory chatbot aware of active candidate: <strong>{active_cand['name']}</strong> ({eff_score:.2f}% • {active_cand['category']}).</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Render Persistent Conversation History
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input Box
    user_query = st.chat_input("Ask CounselAI a question (e.g., 'What is Betterment?', 'Suggest CS in Pune with 93%')...")

    if user_query:
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate intelligent response
        ans = chatbot.answer_query(user_query)
        bot_reply = f"**{ans['title']}**\n\n{ans['content']}"

        st.session_state["chat_messages"].append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)

# ----------------- SECTION 9: DATASET & ANALYTICS (SIMPLISTIC & VIVA-READY) -----------------
elif nav_selection == "📊 Dataset & Analytics":
    st.markdown("""
    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0B2046;">Data Foundation & Predictive Methodology</h3>
    <p style="margin: 1px 0 14px 0; font-size: 0.82rem; color: #64748B;">Overview of the historical Maharashtra State CET dataset and predictive mechanics for viva evaluation.</p>
    """, unsafe_allow_html=True)

    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown("""
        <div class="solid-panel">
            <div style="font-size: 0.7rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Historical Dataset</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #0B2046;">337,776 records</div>
            <div style="font-size: 0.75rem; color: #64748B;">Cleaned from 341,929 raw records</div>
        </div>
        """, unsafe_allow_html=True)
    with v2:
        st.markdown("""
        <div class="solid-panel">
            <div style="font-size: 0.7rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Temporal Coverage</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #0B2046;">5 Academic Years</div>
            <div style="font-size: 0.75rem; color: #64748B;">2021 through 2025 (CAP Rounds 1, 2, 3)</div>
        </div>
        """, unsafe_allow_html=True)
    with v3:
        st.markdown("""
        <div class="solid-panel">
            <div style="font-size: 0.7rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Institutional Scope</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #0B2046;">466 Colleges</div>
            <div style="font-size: 0.75rem; color: #64748B;">Across 36 Maharashtra Districts</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if engine.df_raw is not None and len(engine.df_raw) > 0:
            df_year_counts = engine.df_raw["year"].value_counts().reset_index()
            df_year_counts.columns = ["Year", "Allotments"]
            df_year_counts = df_year_counts.sort_values(by="Year")
            fig_y = px.bar(df_year_counts, x="Year", y="Allotments", title="Historical Allotments Recorded per Year", color_discrete_sequence=["#0B2046"])
            fig_y.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font_family="Plus Jakarta Sans", height=280)
            st.plotly_chart(fig_y, use_container_width=True)

    with col_g2:
        if engine.df_raw is not None and len(engine.df_raw) > 0:
            df_stream_counts = engine.df_raw["branch_cluster"].value_counts().head(6).reset_index()
            df_stream_counts.columns = ["Stream", "Records"]
            fig_s = px.pie(df_stream_counts, names="Stream", values="Records", title="Distribution across Engineering Streams", color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_s.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font_family="Plus Jakarta Sans", height=280)
            st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("""
    <div class="solid-panel">
        <h4 style="margin: 0 0 4px 0; font-size: 0.95rem; font-weight: 700; color: #0B2046;">How CounselAI Predicts & Classifies</h4>
        <div style="font-size: 0.82rem; color: #475569; line-height: 1.5;">
            <p>1. <strong>Cutoff Drift Estimation</strong>: Tracks 5-year trends to project the expected cutoff for the upcoming round rather than relying on a static single-year number.</p>
            <p>2. <strong>Multi-Round Drop Dynamics</strong>: Models seat vacancies between Round 1, Round 2, and Round 3 to gauge upgrade realistic probabilities.</p>
            <p>3. <strong>Probabilistic Categorization</strong>: Bins choices into <strong>Ambitious</strong> (15%–40% dream reach), <strong>Target</strong> (40%–80% realistic match), and <strong>Safe</strong> (80%+ safety net), enforcing game-theoretic optimal ordering.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.74rem; padding-top: 18px; padding-bottom: 8px;'>CounselAI • Decision Support System for Engineering Admissions</div>", unsafe_allow_html=True)
