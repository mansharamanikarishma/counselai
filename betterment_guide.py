"""
Interactive Guidance Module: Self-Freeze vs. Betterment (Float).
Guides students step-by-step through Maharashtra CAP counseling seat decisions,
seat acceptance fees, round progression, and strategic choice-filling pitfalls.
Strictly excludes Auto-Freeze; focuses on active candidate actions.
"""

import streamlit as st

def render_betterment_guide():
    """Renders the comprehensive candidate decision guide in Streamlit."""
    st.header("⚖️ CAP Strategy Guide: Self-Freeze vs. Betterment (Float)")
    st.caption("Understand your official options after seat allotment in CAP Round 1 or Round 2 to maximize your chances.")

    # Two prominent cards comparing the paths
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧊 Option 1: Self-Freeze")
        st.markdown("""
        **When to choose this:**
        You are **completely satisfied** with your allotted college & branch and have no desire to try for higher preferences.

        **What happens:**
        1. You confirm and finalize this seat.
        2. You **pay the ₹1,000 Seat Acceptance Fee** online via the CET portal.
        3. You download your official Allotment Letter.
        4. You report to the college campus within the specified window with your original documents and pay college admission fees.

        ⚠️ **Critical Rule**: Once you Self-Freeze, you **exit the CAP process**. You cannot participate in Round 2 or Round 3!
        """)

    with col2:
        st.subheader("🚀 Option 2: Betterment (Float)")
        st.markdown("""
        **When to choose this:**
        You have received a decent seat, but you want to **aim higher** in the next round without risking what you already have!

        **What happens:**
        1. You accept the seat as your **guaranteed safety net**.
        2. You **pay the ₹1,000 Seat Acceptance Fee** online (only once across all rounds).
        3. You select **'Not Freeze / Betterment'** on the portal.
        4. In Round 2/3, you can rearrange or edit your choice form to aim for higher-preference colleges.

        🛡️ **Safety Guarantee**: If you get a better college in Round 2, your Round 1 seat is released to another student. If you **don't** get an upgrade, your Round 1 seat **remains 100% reserved for you**!
        """)

    st.markdown("---")

    # Interactive Step-by-Step Flowchart
    st.subheader("🗺️ The Golden Decision Flowchart")
    
    tab_r1, tab_r2, tab_mistakes = st.tabs([
        "📍 After Round 1 Allotment", 
        "📍 After Round 2 Allotment", 
        "⚠️ Top 4 Critical Pitfalls to Avoid"
    ])

    with tab_r1:
        st.markdown("""
        #### Round 1 Scenario Guide:
        * **Scenario A: You got a dream college (e.g. COEP / VJTI / PICT Computer Engineering)**
          * *Action*: Choose **Self-Freeze**, pay ₹1,000, report to institute, and celebrate!
        * **Scenario B: You got your 5th or 12th choice (good, but not your #1 preference)**
          * *Action*: Choose **Betterment (Float)**! Pay ₹1,000 seat acceptance fee. You now hold this seat securely in your pocket while competing for choices 1 through 4 in Round 2.
        * **Scenario C: You were NOT allotted any seat in Round 1**
          * *Action*: Don't panic! You don't pay any fee. You automatically move to Round 2. Revise your choice code list to add more **Target** and **Safe** options.
        """)

    with tab_r2:
        st.markdown("""
        #### Round 2 Scenario Guide:
        * **Scenario A: You were upgraded to a higher choice!**
          * *Action*: Your previous Round 1 seat is cancelled. Your new upgraded seat is confirmed! You do not need to pay the ₹1,000 fee again (it carries forward). You can either Self-Freeze now or try for Betterment in Round 3.
        * **Scenario B: No upgrade occurred in Round 2**
          * *Action*: Your Round 1 seat remains **100% safe and intact**. You can either accept it now (Self-Freeze) or carry it into Round 3.
        * **Scenario C: You got allotted a seat for the first time in Round 2**
          * *Action*: Pay the one-time ₹1,000 Seat Acceptance Fee and decide between Self-Freeze and Betterment for Round 3.
        """)

    with tab_mistakes:
        st.markdown("""
        #### 🚫 Rookie Mistakes That Cost Students Their Seats:
        1. **Not paying the ₹1,000 Seat Acceptance Fee after choosing Betterment**:
           * If you choose Betterment on the portal but fail to pay the ₹1,000 fee within the deadline, **your allotted seat is permanently cancelled**, and you are thrown out of CAP Round 2!
        2. **Putting a Safe College at Preference #1**:
           * Never put a backup college at the very top of your option form. Always put your **Ambitious dream colleges first**.
        3. **Reporting in person to college when taking Betterment**:
           * If you chose Betterment, you do **NOT** go to the college physically. You only complete the online process. You only visit the college when you are ready to Freeze and finalize admission.
        4. **Skipping Spot / Institutional Rounds (ACAP)**:
           * Even after Round 3, top colleges have leftover seats (vacated by IIT/NIT aspirants). Always attend on-campus spot rounds!
        """)
