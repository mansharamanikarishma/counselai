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

import importlib
import ml_engine
import counselor_bot
importlib.reload(ml_engine)
importlib.reload(counselor_bot)

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

# Section 8: CAP Round Walkthrough Session State
def walkthrough_reset():
    st.session_state["walkthrough_round"] = 1
    st.session_state["walkthrough_decisions"] = {1: None, 2: None, 3: None}
    st.session_state["walkthrough_allotments"] = {1: None, 2: None, 3: None}
    st.session_state["walkthrough_view"] = "in_progress"
    st.session_state["wt_branches"] = None
    st.session_state["wt_cities"] = None

if "walkthrough_round" not in st.session_state:
    walkthrough_reset()

if "walkthrough_active_candidate_id" not in st.session_state:
    st.session_state["walkthrough_active_candidate_id"] = st.session_state["active_candidate_id"]
elif st.session_state["walkthrough_active_candidate_id"] != st.session_state["active_candidate_id"]:
    st.session_state["walkthrough_active_candidate_id"] = st.session_state["active_candidate_id"]
    walkthrough_reset()

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
def get_ml_engine(_v=4):
    return AdmissionMLEngine()

@st.cache_resource
def get_chatbot(_engine, _v=4):
    return CounselingChatbot(_engine)

engine = get_ml_engine()
chatbot = get_chatbot(engine)

all_cities = engine.get_all_cities()
all_branches = engine.get_all_branches()
colleges_dict = engine.get_all_colleges()

# ----------------- MODAL DIALOG POPUPS (st.dialog) -----------------
@st.dialog("Profile Updated")
def show_profile_updated_dialog(name, score, hu):
    st.success(f"Applicant record for **{name}** has been updated successfully!")
    st.markdown(f"""
    - **Qualifying Score:** `{score:.2f}%`
    - **Home Jurisdiction:** {hu}
    """)
    st.caption("All multi-round prediction models and option forms have been recomputed.")
    if st.button("Continue to Workspace", type="primary", use_container_width=True):
        st.rerun()

@st.dialog("Candidate Enrolled Successfully")
def show_candidate_enrolled_dialog(name, app_id, score):
    st.balloons()
    st.success(f"New applicant file created for **{name}** (`{app_id}`)!")
    st.markdown(f"- **Qualifying Merit Score:** `{score:.2f}%`")
    st.caption("Candidate has been added to the registry and is available in the applicant switcher.")
    if st.button("Open Candidate File", type="primary", use_container_width=True):
        st.rerun()

@st.dialog("Confirm Candidate Removal")
def show_delete_candidate_dialog(cand):
    st.warning(f"Are you sure you want to remove **{cand['name']}** (`{cand['id']}`)?")
    st.write("This will remove all associated option forms and simulation records for this candidate.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with c2:
        if st.button("🗑️ Confirm Remove", type="primary", use_container_width=True):
            st.session_state["candidates"] = [c for c in st.session_state["candidates"] if c["id"] != cand["id"]]
            st.rerun()

@st.dialog("Confirm Self-Freeze Action")
def show_freeze_confirm_dialog(allotment, round_num):
    st.warning("⚠️ **Critical CAP Counseling Notice**")
    st.markdown(f"""
    You are choosing to **Self-Freeze** your seat at:
    ### **{allotment['college_name']}**
    **Branch:** {allotment['branch']} • 📍 {allotment['city']}  
    **DTE College Code:** `{allotment['college_code']}` • **Official Choice Code:** `{allotment['choice_code']}`
    """)
    st.markdown("""
    - Once you Freeze, you **exit further centralized CAP rounds**.
    - You must pay the online ₹1,000 Seat Acceptance Fee on the official portal.
    - You must physically report to the college campus within the reporting window.
    """)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Go Back", use_container_width=True):
            st.rerun()
    with c2:
        if st.button("🧊 Confirm & Freeze Seat", type="primary", use_container_width=True):
            st.session_state["walkthrough_decisions"][round_num] = "Freeze"
            st.session_state["walkthrough_view"] = "summary"
            st.rerun()

@st.dialog("Confirm Betterment (Float) Action")
def show_betterment_confirm_dialog(allotment, round_num):
    st.info("🛡️ **Betterment (Float) Safety Guarantee**")
    st.markdown(f"""
    Your current seat at **{allotment['college_name']}** (*{allotment['branch']}*) is **100% safely reserved**.
    """)
    st.markdown(f"""
    - You are advancing to **Round {round_num + 1}** to explore higher-preference upgrade possibilities.
    - If a higher-preference college drops into your merit range in Round {round_num + 1}, you receive the upgrade.
    - If no upgrade occurs, your Round {round_num} seat remains 100% yours!
    """)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with c2:
        if st.button(f"🚀 Proceed to Round {round_num + 1}", type="primary", use_container_width=True):
            st.session_state["walkthrough_decisions"][round_num] = "Betterment"
            st.session_state["walkthrough_round"] = round_num + 1
            st.rerun()

@st.dialog("🎉 Seat Secured: Admission Confirmed!")
def show_seat_secured_dialog(allotment, round_num):
    st.balloons()
    st.success("### 🎉 Congratulations! Your Engineering Seat is Secured!")
    st.markdown(f"""
    **Allotted Institution:** {allotment['college_name']}  
    **Engineering Stream:** {allotment['branch']}  
    **Institutional Tier:** {allotment['tier']}  
    **DTE College Code:** `{allotment['college_code']}`  
    **Official Choice Code:** `{allotment['choice_code']}`  
    **Final Admission Status:** Confirmed via Round {round_num} CAP  
    """)
    st.markdown("""
    ---
    **Next Mandatory Steps on Official CET Portal:**
    1. Print your Provisional Allotment Letter.
    2. Confirm ₹1,000 online Seat Acceptance receipt is generated.
    3. Visit the allotted college with original documents (10th/12th marksheets, domicile, nationality, category validity if applicable) before the reporting cutoff.
    """)
    if st.button("View Complete Journey Summary", type="primary", use_container_width=True):
        st.session_state["walkthrough_view"] = "summary"
        st.rerun()


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
        st.session_state["walkthrough_active_candidate_id"] = st.session_state["active_candidate_id"]
        walkthrough_reset()
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
            "category": "GUIDED SIMULATION",
            "items": [
                ("🧭 CAP Round Walkthrough", "🧭 CAP Round Walkthrough")
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
                        st.session_state["walkthrough_active_candidate_id"] = cand["id"]
                        walkthrough_reset()
                        st.rerun()
                else:
                    st.button("✓ Live", key=f"cur_{cand['id']}", disabled=True, use_container_width=True)
            with b_del:
                if len(st.session_state["candidates"]) > 1 and not is_active:
                    if st.button("🗑️ Remove", key=f"del_{cand['id']}", use_container_width=True):
                        show_delete_candidate_dialog(cand)

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
                eff = edit_cet if edit_cet is not None else edit_jee
                show_profile_updated_dialog(edit_name, float(eff or 0.0), edit_hu)

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
                    new_eff = new_cet if new_cet is not None else new_jee
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
                    show_candidate_enrolled_dialog(new_name.strip(), new_id, float(new_eff or 0.0))

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
                selected_cities=active_cand["cities"],
                selected_branches=active_cand["branches"]
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

    # Quick Prompt Suggestion Chips
    st.markdown("<div style='font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;'>Suggested Prompts & Instant Inquiries</div>", unsafe_allow_html=True)
    chip_cols = st.columns(4)
    quick_query = None
    with chip_cols[0]:
        if st.button("🎯 Suggest for my score", key="chip_score", use_container_width=True):
            quick_query = f"Suggest best colleges for my score {eff_score:.2f}%"
    with chip_cols[1]:
        if st.button("🏛️ PICT Cutoff & Placement", key="chip_pict", use_container_width=True):
            quick_query = "What is the cutoff and average placement package for PICT Pune?"
    with chip_cols[2]:
        if st.button("⚖️ Compare COEP vs VJTI", key="chip_comp", use_container_width=True):
            quick_query = "Compare COEP vs VJTI"
    with chip_cols[3]:
        if st.button("🧊 Freeze vs Betterment", key="chip_better", use_container_width=True):
            quick_query = "Explain Self-Freeze vs Betterment rules"

    # Render Persistent Conversation History
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input Box
    user_query = st.chat_input("Ask CounselAI a question (e.g., 'What is Betterment?', 'Suggest CS in Pune with 93%')...")

    # If chip clicked or user typed
    effective_query = user_query or quick_query

    if effective_query:
        st.session_state["chat_messages"].append({"role": "user", "content": effective_query})
        with st.chat_message("user"):
            st.markdown(effective_query)

        # Generate intelligent response aware of active candidate profile
        ans = chatbot.answer_query(effective_query, candidate_profile=active_cand)
        bot_reply = f"**{ans['title']}**\n\n{ans['content']}"

        st.session_state["chat_messages"].append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
        st.rerun()

# ----------------- SECTION 8: NEW FEATURE — GUIDED CAP ROUND WALKTHROUGH -----------------
elif nav_selection == "🧭 CAP Round Walkthrough":
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <div>
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0B2046;">CAP Round-by-Round Guided Walkthrough</h3>
            <p style="margin: 1px 0 0 0; font-size: 0.82rem; color: #64748B;">Interactive decision roadmap guiding <strong>{active_cand['name']}</strong> ({eff_score:.2f}% • {active_cand['category']}) through progressive seat allotments and Freeze vs. Betterment choices.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Persistent Disclaimer Banner (Amber Advisory Card)
    st.markdown("""
    <div class="advisory-card" style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #D97706; padding: 12px 16px; margin-bottom: 14px;">
        <div style="display: flex; align-items: flex-start; gap: 10px;">
            <span style="font-size: 1.15rem; line-height: 1;">⚠️</span>
            <div>
                <strong style="font-size: 0.84rem; color: #92400E; display: block; margin-bottom: 2px;">
                    Advisory Simulation Notice & Model Disclaimer
                </strong>
                <span style="font-size: 0.78rem; color: #78350F; line-height: 1.45; display: block;">
                    This guided walkthrough uses the <strong>same historical-data predictive model</strong> as the rest of CounselAI to provide illustrative planning projections. In live Maharashtra CAP counseling, cutoffs vary dynamically each year based on candidate percentiles, quota demands, and seat matrix revisions. This tool is strictly a strategic planning aid — <strong>it does not guarantee seat allotment</strong> and cannot replace the official State CET Cell portal (<a href="https://fe2026.mahacet.org" target="_blank" style="color: #92400E; text-decoration: underline;">mahacet.org</a>) or institutional counseling reporting.
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Step Indicator (Clean Horizontal Progress Element)
    curr_round = st.session_state["walkthrough_round"]
    is_summary = (st.session_state["walkthrough_view"] == "summary")
    
    steps = [
        ("1", "Round 1", "R1 Allotment & Strategy"),
        ("2", "Round 2", "R2 Upgraded Allocation"),
        ("3", "Round 3", "Final CAP Allocation"),
        ("4", "Summary", "Decision Recap & Actions")
    ]
    
    step_cols = st.columns(4)
    for idx, (s_num, s_title, s_desc) in enumerate(steps):
        step_idx = idx + 1
        is_current = (is_summary and step_idx == 4) or (not is_summary and curr_round == step_idx)
        is_completed = (step_idx < curr_round) or (is_summary and step_idx < 4)
        
        if is_current:
            border_style = "1px solid #DBEAFE"
            border_left = "3.5px solid #2563EB"
            bg_style = "#EFF6FF"
            title_color = "#1E40AF"
            badge_icon = "● Active"
            badge_style = "background: #DBEAFE; color: #1E40AF; font-size: 0.62rem; font-weight: 700; padding: 1px 6px; border-radius: 6px;"
        elif is_completed:
            border_style = "1px solid #BBF7D0"
            border_left = "3.5px solid #16A34A"
            bg_style = "#F0FDF4"
            title_color = "#166534"
            decision_tag = st.session_state["walkthrough_decisions"].get(step_idx)
            badge_icon = f"✓ {decision_tag}" if decision_tag else "✓ Done"
            badge_style = "background: #DCFCE7; color: #166534; font-size: 0.62rem; font-weight: 700; padding: 1px 6px; border-radius: 6px;"
        else:
            border_style = "1px solid #E2E8F0"
            border_left = "1px solid #E2E8F0"
            bg_style = "#FFFFFF"
            title_color = "#64748B"
            badge_icon = "Upcoming"
            badge_style = "background: #F1F5F9; color: #64748B; font-size: 0.62rem; font-weight: 600; padding: 1px 6px; border-radius: 6px;"

        with step_cols[idx]:
            st.markdown(f"""
            <div style="background: {bg_style}; border: {border_style}; border-left: {border_left}; border-radius: 6px; padding: 8px 10px; margin-bottom: 12px; min-height: 64px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                    <span style="font-size: 0.78rem; font-weight: 700; color: {title_color};">{s_title}</span>
                    <span style="{badge_style}">{badge_icon}</span>
                </div>
                <div style="font-size: 0.7rem; color: #64748B;">{s_desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # 3. View Switch: Summary or In-Progress
    if is_summary:
        st.markdown("<div style='font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: #0B2046; letter-spacing: 0.5px; margin-bottom: 8px;'>Counseling Simulation Summary & Record</div>", unsafe_allow_html=True)
        
        recap_records = []
        for r_num in [1, 2, 3]:
            decision = st.session_state["walkthrough_decisions"].get(r_num)
            allotment = st.session_state["walkthrough_allotments"].get(r_num)
            if decision or allotment:
                recap_records.append({
                    "CAP Round": f"Round {r_num}",
                    "Projected College": allotment["college_name"] if allotment else "No Seat Allotted",
                    "Branch": allotment["branch"] if allotment else "-",
                    "City": allotment["city"] if allotment else "-",
                    "Projected Cutoff": f"{allotment['predicted_cutoff']}%" if allotment else "-",
                    "Suitability": f"{allotment['suitability_pct']}%" if allotment else "-",
                    "Action Simulated": "🧊 Self-Freeze" if decision == "Freeze" else ("🚀 Betterment (Float)" if decision == "Betterment" else "Completed Round")
                })

        if recap_records:
            st.dataframe(pd.DataFrame(recap_records), use_container_width=True)

        final_round = max([r for r in [1, 2, 3] if st.session_state["walkthrough_decisions"].get(r) is not None], default=1)
        final_decision = st.session_state["walkthrough_decisions"].get(final_round)
        final_allotment = st.session_state["walkthrough_allotments"].get(final_round)

        if final_decision == "Freeze":
            c_name = final_allotment['college_name'] if final_allotment else 'your allotted institution'
            b_name = final_allotment['branch'] if final_allotment else ''
            st.markdown(f"""
            <div class="solid-panel" style="border-left: 4px solid #166534; background: #F0FDF4; margin-top: 12px;">
                <div style="font-weight: 700; font-size: 0.92rem; color: #166534; margin-bottom: 4px;">
                    Simulation Outcome: Seat Confirmed via Self-Freeze in Round {final_round}
                </div>
                <div style="font-size: 0.8rem; color: #14532D; line-height: 1.5;">
                    By choosing <strong>Self-Freeze</strong>, you confirmed your seat at <strong>{c_name}</strong> ({b_name}).
                    <br>• <strong>Mandatory Portal Action:</strong> Pay the online ₹1,000 Seat Acceptance Fee on the official portal.
                    <br>• <strong>Physical Reporting:</strong> Report to the allotted institute with your original document dossier before the reporting deadline.
                    <br>• <strong>CAP Status:</strong> You permanently exit further centralized CAP rounds.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            c_name = final_allotment['college_name'] if final_allotment else 'your final allotted college'
            b_name = final_allotment['branch'] if final_allotment else ''
            st.markdown(f"""
            <div class="solid-panel" style="border-left: 4px solid #2563EB; background: #EFF6FF; margin-top: 12px;">
                <div style="font-weight: 700; font-size: 0.92rem; color: #1E40AF; margin-bottom: 4px;">
                    Simulation Outcome: Round 3 Final Allocation Reached
                </div>
                <div style="font-size: 0.8rem; color: #1E3A8A; line-height: 1.5;">
                    You completed the multi-round Betterment progression through Round 3. Your retained or upgraded seat at <strong>{c_name}</strong> ({b_name}) represents your final centralized allotment.
                    <br>• <strong>Final Acceptance:</strong> Complete institutional reporting with the ₹1,000 Seat Acceptance receipt and original certificates.
                    <br>• <strong>Institutional Spot Rounds (ACAP):</strong> If you still wish to explore vacant seats at Tier-1 colleges, participate in on-campus spot rounds post-CAP.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        c_btn1, c_btn2 = st.columns([1, 2])
        with c_btn1:
            if st.button("🔄 Restart Walkthrough", type="primary", use_container_width=True):
                walkthrough_reset()
                st.rerun()
        with c_btn2:
            if final_allotment:
                if st.button("📜 View Official Admission Slip", use_container_width=True):
                    show_seat_secured_dialog(final_allotment, final_round)

    else:
        # =========================================================================
        # ROUND 1: INTERACTIVE ALLOTMENT SELECTION & FILTERS
        # =========================================================================
        if curr_round == 1:
            st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #0B2046; margin-bottom: 4px;'>Round 1: Initial Allotment & Decision Center</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 0.8rem; color: #64748B; margin-bottom: 12px;'>Explore eligible colleges matching your percentile (<strong>{eff_score:.2f}%</strong> • {active_cand['category']}). Filter, inspect options, and select which college you want to simulate as your allotted seat.</div>", unsafe_allow_html=True)

            # Interactive Filter Bar
            with st.container(border=True):
                st.markdown("<div style='font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;'>🎯 Filter & Explore Round 1 College Options</div>", unsafe_allow_html=True)
                f1, f2, f3 = st.columns([2, 2, 1.5])
                with f1:
                    wt_cities = st.multiselect("Preferred Cities", options=all_cities, default=active_cand["cities"], key="wt_r1_cities")
                with f2:
                    wt_branches = st.multiselect("Engineering Streams", options=all_branches, default=active_cand["branches"], key="wt_r1_branches")
                with f3:
                    wt_tier = st.selectbox("Category Band", ["All Matches", "Target & Safe Only", "Ambitious Only"], index=0, key="wt_r1_tier")

            # Persist Round 1 filters into session state for downstream rounds
            st.session_state["wt_branches"] = wt_branches if wt_branches else active_cand.get("branches")
            st.session_state["wt_cities"] = wt_cities if wt_cities else active_cand.get("cities")

            # Predict Round 1 options using filters
            rnd_preds = engine.predict_choices(
                score_cet=active_cand.get("score_cet"),
                score_jee=active_cand.get("score_jee"),
                exam_mode=active_cand["exam_mode"],
                category=active_cand["category"],
                gender=active_cand["gender"],
                selected_cities=st.session_state["wt_cities"],
                selected_branches=st.session_state["wt_branches"],
                round_target="Round 1"
            )

            all_choices = rnd_preds["all_ordered"]
            if wt_tier == "Target & Safe Only":
                filtered_choices = all_choices[all_choices["category_tag"].isin(["Target", "Safe"])]
                if len(filtered_choices) > 0:
                    all_choices = filtered_choices
            elif wt_tier == "Ambitious Only":
                filtered_choices = all_choices[all_choices["category_tag"] == "Ambitious"]
                if len(filtered_choices) > 0:
                    all_choices = filtered_choices

            if len(all_choices) == 0:
                all_choices = rnd_preds["all_ordered"]

            if len(all_choices) == 0:
                st.warning("No colleges found matching these specific filters. Try expanding your city or branch selection.")
            else:
                # Active Held Seat in Round 1
                curr_held = st.session_state["walkthrough_allotments"].get(1)
                if curr_held is None:
                    # Default to top Target/Safe match
                    best_matches = all_choices[all_choices["category_tag"].isin(["Target", "Safe"])]
                    if len(best_matches) > 0:
                        curr_held = best_matches.iloc[0].to_dict()
                    else:
                        curr_held = all_choices.iloc[0].to_dict()
                    st.session_state["walkthrough_allotments"][1] = curr_held

                # Dropdown & Quick Selection Cards for Candidates
                st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #0B2046; margin: 14px 0 6px 0;'>Select Your Simulated Allotted College for Round 1:</div>", unsafe_allow_html=True)
                
                # Selection Dropdown
                options_list = all_choices.head(20).to_dict("records")
                option_labels = [f"{r['college_name']} — {r['branch']} ({r['predicted_cutoff']}% • {r['category_tag']})" for r in options_list]
                
                current_label_idx = 0
                for idx, r in enumerate(options_list):
                    if r["college_code"] == curr_held["college_code"] and r["branch"] == curr_held["branch"]:
                        current_label_idx = idx
                        break

                sel_idx = st.selectbox(
                    "Pick from available matching colleges:",
                    options=range(len(option_labels)),
                    format_func=lambda i: option_labels[i],
                    index=current_label_idx,
                    key="r1_sel_dropdown"
                )

                if options_list[sel_idx]["college_code"] != curr_held["college_code"] or options_list[sel_idx]["branch"] != curr_held["branch"]:
                    st.session_state["walkthrough_allotments"][1] = options_list[sel_idx]
                    curr_held = options_list[sel_idx]
                    st.rerun()

                # Quick Choice Cards: Compact Horizontal Tiles 1 to 4
                st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #0B2046; text-transform: uppercase; margin: 12px 0 8px 0; letter-spacing: 0.4px;'>Recommended Allotment Options (Horizontal Tiles 1–4):</div>", unsafe_allow_html=True)
                for c_idx in range(min(len(options_list), 4)):
                    opt = options_list[c_idx]
                    is_active_opt = (opt["college_code"] == curr_held["college_code"] and opt["branch"] == curr_held["branch"])
                    tile_num = c_idx + 1
                    tag_color = "#16A34A" if opt.get("category_tag") == "Safe" else ("#D97706" if opt.get("category_tag") == "Target" else "#DC2626")
                    tag_bg = "#DCFCE7" if opt.get("category_tag") == "Safe" else ("#FEF3C7" if opt.get("category_tag") == "Target" else "#FEE2E2")
                    with st.container(border=True):
                        c_info, c_meta, c_btn = st.columns([5.5, 2.7, 1.8], vertical_alignment="center")
                        with c_info:
                            st.markdown(f"""
                            <div style="line-height: 1.35;">
                                <div style="display: flex; align-items: center; gap: 6px;">
                                    <span style="background: #DBEAFE; color: #1D4ED8; font-size: 0.72rem; font-weight: 700; padding: 1px 6px; border-radius: 4px; white-space: nowrap;">Tile {tile_num}</span>
                                    <span style="font-weight: 700; font-size: 0.88rem; color: #0B2046;">{opt['college_name']}</span>
                                </div>
                                <div style="font-size: 0.79rem; color: #334155; margin-top: 2px;">
                                    <strong>{opt['branch']}</strong> <span style="color: #64748B;">• 📍 {opt['city']} • {opt.get('tier', 'Tier-2')} • DTE: <code>{opt['college_code']}</code></span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        with c_meta:
                            st.markdown(f"""
                            <div style="text-align: right; line-height: 1.35;">
                                <span style="font-size: 0.7rem; color: #64748B;">Projected Cutoff:</span> <strong style="font-size: 0.92rem; color: #0B2046;">{opt['predicted_cutoff']}%</strong><br>
                                <span style="font-size: 0.7rem; color: #16A34A; font-weight: 600;">{opt['suitability_pct']}% Match</span> • <span style="font-size: 0.7rem; background: {tag_bg}; color: {tag_color}; font-weight: 700; padding: 1px 5px; border-radius: 3px;">{opt.get('category_tag', 'Target')}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        with c_btn:
                            if is_active_opt:
                                st.button("✓ Allotted", key=f"btn_r1_sel_{c_idx}", disabled=True, use_container_width=True)
                            else:
                                if st.button(f"👉 Select Tile {tile_num}", key=f"btn_pick_r1_{c_idx}", use_container_width=True, type="primary"):
                                    st.session_state["walkthrough_allotments"][1] = opt
                                    st.rerun()

                # Prominently Display Current Allotted Seat Card
                b_class = "badge-safe" if curr_held.get("category_tag") == "Safe" else ("badge-target" if curr_held.get("category_tag") == "Target" else "badge-ambitious")
                
                st.markdown(f"""
                <div class="advisory-card" style="border: 2px solid #2563EB; background: #F8FAFC; padding: 14px 18px; margin: 12px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
                        <div>
                            <span style="background: #2563EB; color: #FFFFFF; font-size: 0.68rem; font-weight: 700; padding: 3px 8px; border-radius: 4px; text-transform: uppercase;">
                                🔒 Current Allotted Seat (Round 1)
                            </span>
                            <span class="{b_class}" style="margin-left: 6px;">{curr_held.get('badge', curr_held.get('category_tag', 'Target'))}</span>
                            <span style="font-size: 0.74rem; color: #64748B; margin-left: 8px;">DTE: <code>{curr_held['college_code']}</code></span>
                            <span style="font-size: 0.74rem; color: #64748B; margin-left: 8px;">Choice Code: <code>{curr_held.get('choice_code', curr_held['college_code'] + '101')}</code></span>
                            <h4 style="margin: 6px 0 2px 0; font-size: 1.08rem; font-weight: 700; color: #0B2046;">
                                {curr_held['college_name']}
                            </h4>
                            <div style="font-size: 0.88rem; color: #334155;">
                                <strong>{curr_held['branch']}</strong> • 📍 {curr_held['city']} • {curr_held['tier']}
                            </div>
                        </div>
                        <div style="text-align: right; min-width: 120px;">
                            <div style="font-size: 0.65rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Projected Cutoff</div>
                            <div style="font-size: 1.4rem; font-weight: 800; color: #0B2046;">{curr_held.get('predicted_cutoff', curr_held.get('r1_cutoff', 90.0))}%</div>
                            <div style="font-size: 0.72rem; color: #16A34A; font-weight: 600;">Suitability: {curr_held.get('suitability_pct', 85)}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Strategic Decision Actions for Round 1
                st.markdown("""
                <div class="solid-panel" style="padding: 12px 16px; margin-top: 10px; margin-bottom: 12px;">
                    <div style="font-size: 0.84rem; font-weight: 700; color: #0B2046;">Simulate Your Round 1 Counseling Decision:</div>
                    <div style="font-size: 0.76rem; color: #64748B; margin-top: 2px;">
                        • <strong>Self-Freeze:</strong> Confirms this seat permanently and exits centralized counseling.<br>
                        • <strong>Betterment (Float):</strong> Guarantees this seat remains 100% reserved while exploring higher-tier upgrades in Round 2.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                d1, d2 = st.columns(2)
                with d1:
                    if st.button("🧊 Simulate: Self-Freeze Seat", use_container_width=True, type="secondary"):
                        show_freeze_confirm_dialog(curr_held, 1)
                with d2:
                    if st.button("🚀 Simulate: Betterment (Float) & Advance to Round 2", use_container_width=True, type="primary"):
                        show_betterment_confirm_dialog(curr_held, 1)

        # =========================================================================
        # ROUND 2: BETTERMENT UPGRADE EVALUATION & DECISION
        # =========================================================================
        elif curr_round == 2:
            r1_seat = st.session_state["walkthrough_allotments"].get(1)
            if not r1_seat:
                r1_seat = all_df.iloc[0].to_dict()
                st.session_state["walkthrough_allotments"][1] = r1_seat

            st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #0B2046; margin-bottom: 4px;'>Round 2: Betterment Upgrades & Vacancy Allocation</div>", unsafe_allow_html=True)
            
            # Safety Net Guarantee Card
            st.markdown(f"""
            <div style="background: #ECFDF5; border: 1.5px solid #10B981; border-radius: 7px; padding: 10px 14px; margin-bottom: 14px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.1rem;">🛡️</span>
                    <div>
                        <strong style="color: #065F46; font-size: 0.84rem;">Betterment Safety Net Active:</strong>
                        <span style="color: #047857; font-size: 0.8rem;">
                            Your Round 1 seat at <strong>{r1_seat['college_name']}</strong> ({r1_seat['branch']}) is <strong>100% reserved</strong>. You cannot lose this seat unless you choose to accept an upgraded allocation.
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            pref_branches = st.session_state.get("wt_branches") or active_cand.get("branches")
            pref_cities = st.session_state.get("wt_cities") or active_cand.get("cities")

            # Query Betterment Upgrade Opportunities
            with st.spinner("Analyzing Round 2 cutoff drift and vacancy movements..."):
                upgrades_df = engine.simulate_betterment(
                    allotted_college_code=r1_seat["college_code"],
                    allotted_branch=r1_seat["branch"],
                    student_score=eff_score,
                    category=active_cand["category"],
                    selected_cities=pref_cities,
                    selected_branches=pref_branches
                )

            # Also pull Round 2 predicted choices for active candidate
            r2_preds = engine.predict_choices(
                score_cet=active_cand.get("score_cet"),
                score_jee=active_cand.get("score_jee"),
                exam_mode=active_cand["exam_mode"],
                category=active_cand["category"],
                gender=active_cand["gender"],
                selected_cities=pref_cities,
                selected_branches=pref_branches,
                round_target="Round 2"
            )["all_ordered"]

            # Combine or display upgrade candidates (ensuring at least 4 options)
            available_upgrades = []
            if len(upgrades_df) > 0:
                available_upgrades = upgrades_df.to_dict("records")

            # Pad up to 4 options using r2_preds if simulate_betterment returned fewer than 4
            already_keys = set((u.get("college_code"), u.get("branch")) for u in available_upgrades)
            already_keys.add((r1_seat.get("college_code"), r1_seat.get("branch")))

            for _, r in r2_preds.iterrows():
                if len(available_upgrades) >= 4:
                    break
                key = (r["college_code"], r["branch"])
                if key not in already_keys:
                    available_upgrades.append({
                        "college_code": r["college_code"],
                        "college_name": r["college_name"],
                        "branch": r["branch"],
                        "city": r["city"],
                        "tier": r["tier"],
                        "avg_placement": r["avg_placement"],
                        "r2_cutoff": r["predicted_cutoff"],
                        "predicted_cutoff": r["predicted_cutoff"],
                        "upgrade_chance_pct": r["suitability_pct"]
                    })
                    already_keys.add(key)

            st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #0B2046; margin: 12px 0 8px 0;'>✨ Upgraded Seats Available in Round 2 (Horizontal Tiles 1–4):</div>", unsafe_allow_html=True)

            # Current seat for Round 2 (defaults to R1 if no upgrade chosen yet)
            r2_current = st.session_state["walkthrough_allotments"].get(2) or r1_seat

            # Render 4 upgrade choices as compact horizontal tiles
            for idx in range(min(len(available_upgrades), 4)):
                upg = available_upgrades[idx]
                u_code = upg.get("college_code", "0000")
                u_branch = upg.get("branch", "Computer Engineering")
                u_name = upg.get("college_name", "College")
                u_tier = upg.get("tier", "Tier-2")
                u_ctc = upg.get("avg_placement", "₹8.5 LPA")
                u_chance = upg.get("upgrade_chance_pct", 75)
                u_r2_cutoff = upg.get("r2_cutoff", upg.get("predicted_cutoff", 92.5))
                u_city = upg.get("city", pref_cities[0] if pref_cities else "Pune")
                
                is_this_selected = (r2_current.get("college_code") == u_code and r2_current.get("branch") == u_branch)
                tile_num = idx + 1

                with st.container(border=True):
                    c_info, c_meta, c_btn = st.columns([5.5, 2.7, 1.8], vertical_alignment="center")
                    with c_info:
                        st.markdown(f"""
                        <div style="line-height: 1.35;">
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <span style="background: #DCFCE7; color: #166534; font-size: 0.72rem; font-weight: 700; padding: 1px 6px; border-radius: 4px; white-space: nowrap;">Tile {tile_num}</span>
                                <span style="font-weight: 700; font-size: 0.88rem; color: #0B2046;">{u_name}</span>
                            </div>
                            <div style="font-size: 0.79rem; color: #334155; margin-top: 2px;">
                                <strong>{u_branch}</strong> <span style="color: #64748B;">• 📍 {u_city} • {u_tier} • DTE: <code>{u_code}</code></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c_meta:
                        st.markdown(f"""
                        <div style="text-align: right; line-height: 1.35;">
                            <span style="font-size: 0.7rem; color: #64748B;">Round 2 Cutoff:</span> <strong style="font-size: 0.92rem; color: #0B2046;">{u_r2_cutoff}%</strong><br>
                            <span style="font-size: 0.7rem; color: #16A34A; font-weight: 700;">🚀 {u_chance}% Upgrade Chance</span> <span style="color: #94A3B8;">•</span> <span style="font-size: 0.7rem; color: #475569;">{u_ctc}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with c_btn:
                        if is_this_selected:
                            st.button("✓ Selected", key=f"btn_upg_sel_{idx}", disabled=True, use_container_width=True)
                        else:
                            if st.button(f"✨ Accept Tile {tile_num}", key=f"btn_accept_upg_{idx}", use_container_width=True, type="primary"):
                                st.session_state["walkthrough_allotments"][2] = {
                                    "college_code": u_code,
                                    "choice_code": f"{u_code}101",
                                    "college_name": u_name,
                                    "branch": u_branch,
                                    "city": u_city,
                                    "tier": u_tier,
                                    "predicted_cutoff": u_r2_cutoff,
                                    "suitability_pct": u_chance,
                                    "category_tag": "Target"
                                }
                                st.rerun()

            # Option to explicitly keep R1 seat
            if r2_current.get("college_code") != r1_seat["college_code"]:
                if st.button(f"🛡️ Revert & Retain Original Round 1 Seat ({r1_seat['college_name'][:35]}...)", use_container_width=True):
                    st.session_state["walkthrough_allotments"][2] = r1_seat
                    st.rerun()

            # Display Currently Held Seat in Round 2
            st.markdown(f"""
            <div class="advisory-card" style="border: 2px solid #0B2046; background: #FFFFFF; padding: 12px 16px; margin: 12px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="background: #0B2046; color: white; font-size: 0.65rem; font-weight: 700; padding: 2px 6px; border-radius: 4px;">
                            CURRENT HELD SEAT FOR ROUND 2
                        </span>
                        <h4 style="margin: 6px 0 2px 0; font-size: 1.05rem; font-weight: 700; color: #0B2046;">{r2_current['college_name']}</h4>
                        <div style="font-size: 0.84rem; color: #475569;"><strong>{r2_current['branch']}</strong> • 📍 {r2_current.get('city', '-')} • {r2_current.get('tier', 'Tier-2')}</div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.72rem; color: #64748B;">Cutoff:</span>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #0B2046;">{r2_current.get('predicted_cutoff', 90.0)}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            d2_1, d2_2 = st.columns(2)
            with d2_1:
                if st.button("🧊 Simulate: Self-Freeze (Confirm This Seat Now)", use_container_width=True, type="secondary"):
                    show_freeze_confirm_dialog(r2_current, 2)
            with d2_2:
                if st.button("🚀 Simulate: Betterment (Float) to Round 3 (Final Round)", use_container_width=True, type="primary"):
                    show_betterment_confirm_dialog(r2_current, 2)

        # =========================================================================
        # ROUND 3: FINAL ROUND ALLOTMENT & SEAT SECURED
        # =========================================================================
        elif curr_round == 3:
            # Active held seat from Round 2 or Round 1
            r3_seat = st.session_state["walkthrough_allotments"].get(2) or st.session_state["walkthrough_allotments"].get(1)
            if not r3_seat:
                r3_seat = all_df.iloc[0].to_dict()
                st.session_state["walkthrough_allotments"][3] = r3_seat

            st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #0B2046; margin-bottom: 4px;'>Round 3: Final Centralized Allotment & Admission Confirmation</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: #EFF6FF; border: 1.5px solid #3B82F6; border-radius: 7px; padding: 10px 14px; margin-bottom: 14px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.1rem;">🏁</span>
                    <div>
                        <strong style="color: #1E40AF; font-size: 0.84rem;">Final Centralized Round:</strong>
                        <span style="color: #1E3A8A; font-size: 0.8rem;">
                            Round 3 concludes the centralized CAP process. All remaining institutional vacancies are definitively filled.
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            pref_branches = st.session_state.get("wt_branches") or active_cand.get("branches")
            pref_cities = st.session_state.get("wt_cities") or active_cand.get("cities")

            # Query Round 3 options to provide 4 tiles
            r3_preds = engine.predict_choices(
                score_cet=active_cand.get("score_cet"),
                score_jee=active_cand.get("score_jee"),
                exam_mode=active_cand["exam_mode"],
                category=active_cand["category"],
                gender=active_cand["gender"],
                selected_cities=pref_cities,
                selected_branches=pref_branches,
                round_target="Round 3"
            )["all_ordered"]

            r3_options = [r3_seat]
            already_keys = {(r3_seat.get("college_code"), r3_seat.get("branch"))}
            for _, r in r3_preds.iterrows():
                if len(r3_options) >= 4:
                    break
                key = (r["college_code"], r["branch"])
                if key not in already_keys:
                    r3_options.append({
                        "college_code": r["college_code"],
                        "choice_code": r.get("choice_code", f"{r['college_code']}101"),
                        "college_name": r["college_name"],
                        "branch": r["branch"],
                        "city": r["city"],
                        "tier": r["tier"],
                        "predicted_cutoff": r["predicted_cutoff"],
                        "suitability_pct": r["suitability_pct"],
                        "category_tag": r["category_tag"]
                    })
                    already_keys.add(key)

            st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #0B2046; margin: 12px 0 8px 0;'>🎯 Final Round Allocation Options (Horizontal Tiles 1–4):</div>", unsafe_allow_html=True)
            for idx in range(min(len(r3_options), 4)):
                opt = r3_options[idx]
                is_selected = (r3_seat.get("college_code") == opt.get("college_code") and r3_seat.get("branch") == opt.get("branch"))
                tile_num = idx + 1
                with st.container(border=True):
                    c_info, c_meta, c_btn = st.columns([5.5, 2.7, 1.8], vertical_alignment="center")
                    with c_info:
                        st.markdown(f"""
                        <div style="line-height: 1.35;">
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <span style="background: #FEF3C7; color: #92400E; font-size: 0.72rem; font-weight: 700; padding: 1px 6px; border-radius: 4px; white-space: nowrap;">Tile {tile_num}</span>
                                <span style="font-weight: 700; font-size: 0.88rem; color: #0B2046;">{opt['college_name']}</span>
                            </div>
                            <div style="font-size: 0.79rem; color: #334155; margin-top: 2px;">
                                <strong>{opt['branch']}</strong> <span style="color: #64748B;">• 📍 {opt.get('city', '-')} • {opt.get('tier', 'Tier-2')} • DTE: <code>{opt['college_code']}</code></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c_meta:
                        badge_label = "Confirmed Allotment" if is_selected else f"{opt.get('category_tag', 'Target')} Option"
                        badge_bg = "#DCFCE7" if is_selected else "#FEF3C7"
                        badge_color = "#166534" if is_selected else "#92400E"
                        st.markdown(f"""
                        <div style="text-align: right; line-height: 1.35;">
                            <span style="font-size: 0.7rem; color: #64748B;">Final Cutoff:</span> <strong style="font-size: 0.92rem; color: #0B2046;">{opt.get('predicted_cutoff', 90.0)}%</strong><br>
                            <span style="font-size: 0.7rem; background: {badge_bg}; color: {badge_color}; font-weight: 700; padding: 1px 5px; border-radius: 3px;">● {badge_label}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with c_btn:
                        if is_selected:
                            st.button("✓ Confirmed", key=f"btn_r3_sel_{idx}", disabled=True, use_container_width=True)
                        else:
                            if st.button(f"👉 Select Tile {tile_num}", key=f"btn_r3_pick_{idx}", use_container_width=True, type="primary"):
                                st.session_state["walkthrough_allotments"][3] = opt
                                st.rerun()

            # Final Allotment Display Card
            st.markdown(f"""
            <div class="advisory-card" style="border: 2.5px solid #16A34A; background: #F0FDF4; padding: 16px 20px; margin: 14px 0;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <span style="background: #16A34A; color: white; font-size: 0.68rem; font-weight: 700; padding: 3px 8px; border-radius: 4px; text-transform: uppercase;">
                            🎯 FINAL ALLOTTED SEAT FOR ADMISSION
                        </span>
                        <h3 style="margin: 8px 0 3px 0; font-size: 1.2rem; font-weight: 800; color: #0B2046;">
                            {r3_seat['college_name']}
                        </h3>
                        <div style="font-size: 0.92rem; color: #166534; font-weight: 600;">
                            {r3_seat['branch']} • 📍 {r3_seat.get('city', '-')} • {r3_seat.get('tier', 'Tier-2')}
                        </div>
                        <div style="font-size: 0.78rem; color: #475569; margin-top: 6px;">
                            DTE College Code: <code>{r3_seat.get('college_code', '0000')}</code> • Official Choice Code: <code>{r3_seat.get('choice_code', '0000101')}</code>
                        </div>
                    </div>
                    <div style="text-align: right; min-width: 120px;">
                        <div style="font-size: 0.65rem; text-transform: uppercase; color: #166534; font-weight: 700;">Final Cutoff</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #166534;">{r3_seat.get('predicted_cutoff', 90.0)}%</div>
                        <div style="font-size: 0.72rem; color: #15803D; font-weight: 700;">Status: Ready to Secure</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.session_state["walkthrough_allotments"][3] = r3_seat

            # Finalize & Secure Admission Button
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            if st.button("🎓 Finalize Admission & Secure Seat!", type="primary", use_container_width=True):
                st.session_state["walkthrough_decisions"][3] = "Admission Secured"
                show_seat_secured_dialog(r3_seat, 3)

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

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Master Dataset Download Card
    with st.container(border=True):
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div>
                <h4 style="margin: 0; font-size: 1rem; font-weight: 700; color: #0B2046;">📥 Consolidated Master CAP Dataset (2021–2025)</h4>
                <p style="margin: 2px 0 0 0; font-size: 0.8rem; color: #64748B;">
                    Reconciled 341,929 rows with 100% verified 4-digit DTE codes and official 9-digit choice codes.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        import os
        csv_candidates = [
            "data/unified_cap_data.csv",
            "C:\\Users\\Karishma\\.gemini\\antigravity\\scratch\\mhtcet_predictor\\data\\unified_cap_data.csv",
            "C:\\Users\\Karishma\\OneDrive\\Documents\\College\\Semester 3\\MinorPR\\CounselAI\\data\\unified_cap_data.csv"
        ]
        found_csv = None
        for p in csv_candidates:
            if os.path.exists(p):
                found_csv = p
                break

        if found_csv:
            with open(found_csv, "rb") as f_csv:
                st.download_button(
                    label="⬇️ Download unified_cap_data.csv (62.8 MB)",
                    data=f_csv,
                    file_name="unified_cap_data.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
            st.caption(f"📁 Local File Path: `{found_csv}`")
        else:
            st.warning("Master CSV not found at default data paths.")

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
