"""
Smart Offline AI Counseling Assistant for MHT-CET & JEE Main Candidates.
Dual-Functionality:
1. Answers CAP counseling rules, document requirements, TFWS, and Betterment/Freeze questions.
2. Natural language college query intent matching (e.g., 'I scored 94 in CET, suggest IT colleges in Pune').
Zero external API tokens consumed — 100% local rule & semantic intent matching.
"""

import re
from typing import Dict, Any, List, Optional
from data_processor import COLLEGE_TIER_DATABASE, DEFAULT_TIER

COUNSELING_KNOWLEDGE_BASE = [
    {
        "keywords": ["betterment", "float", "upgrade", "not freeze"],
        "title": "What is Betterment (Float) and how do I choose it?",
        "answer": """
**Betterment (Float) Explained:**
- **What it means**: If you are allotted a college in Round 1 or Round 2 that is NOT your top preference, you can accept the seat as a safety net while keeping your options open for higher-preference colleges in subsequent rounds.
- **How to choose**:
  1. Log into the official CET CAP portal.
  2. Select **'Not Freeze / Betterment'**.
  3. Pay the one-time **Seat Acceptance Fee of ₹1,000** online.
- **The Golden Rule**: If you get upgraded to a better choice in Round 2, your Round 1 seat is automatically cancelled and given to someone else. If you do NOT get an upgrade, your Round 1 seat **remains 100% reserved for you**!
        """
    },
    {
        "keywords": ["freeze", "self freeze", "accept seat", "confirm admission"],
        "title": "When and how should I choose Self-Freeze?",
        "answer": """
**Self-Freeze Explained:**
- **What it means**: You are completely satisfied with your allotted seat and DO NOT want to participate in subsequent CAP rounds.
- **Action steps**:
  1. Log into the CET portal and choose **'Self-Freeze'**.
  2. Pay the **₹1,000 Seat Acceptance Fee** online.
  3. Download your Allotment Letter and Seat Acceptance Receipt.
  4. Report to the allotted college within the specified dates with your original documents and admission fees.
- **Warning**: Once you Freeze, you exit the CAP counseling process—you cannot apply for higher preferences in Round 2 or 3!
        """
    },
    {
        "keywords": ["tfws", "tuition fee waiver", "fee waiver", "income limit"],
        "title": "What is TFWS (Tuition Fee Waiver Scheme)?",
        "answer": """
**Tuition Fee Waiver Scheme (TFWS):**
- **Seat Allocation**: 5% extra supernumerary seats in every branch of every engineering college in Maharashtra.
- **Benefit**: 100% Tuition Fee waiver throughout all 4 years of B.Tech (you only pay Development Fees & Exam Fees).
- **Eligibility**:
  - Annual family income must be **less than ₹8,00,000**.
  - Must upload a valid Income Certificate from the Competent Authority (Tehsildar).
- **Important Note**: TFWS cutoff percentiles are typically 1% to 3% HIGHER than open category cutoffs for the same branch because seats are limited!
        """
    },
    {
        "keywords": ["document", "documents", "certificate", "validity", "creamy layer", "ncl"],
        "title": "Mandatory Documents for Maharashtra CAP Admission",
        "answer": """
**Essential Document Checklist:**
1. **All Candidates**:
   - MHT-CET / JEE Main Scorecard
   - SSC (10th) & HSC (12th) Marksheets
   - Domicile Certificate & Nationality Certificate (or Birth Certificate mentioning birthplace in Maharashtra)
   - Transfer / Leaving Certificate (TC/LC)
2. **Reserved Categories (OBC / VJ-NT / SBC)**:
   - Caste Certificate
   - **Caste Validity Certificate (CVC)** (Compulsory at reporting time!)
   - **Non-Creamy Layer Certificate (NCL)** valid up to March 31 of current fiscal year.
3. **EWS Candidates**:
   - Economically Weaker Section (EWS) Eligibility Certificate issued by competent state authority.
        """
    },
    {
        "keywords": ["home university", "hu", "ohu", "other than home university", "state level"],
        "title": "Home University (HU) vs Other Than Home University (OHU)",
        "answer": """
**University Area Seat Distribution:**
- **Home University (HU)**: The university jurisdiction where you passed your 12th standard (HSC). E.g., if you passed HSC in Pune, SPPU (Pune University) is your HU.
- **Quota Split**: In non-autonomous aided/unaided colleges, ~70% of seats are reserved for HU candidates and ~30% for OHU candidates.
- **Cutoff Difference**: HU cutoffs are generally slightly lower (easier to get) than OHU cutoffs for the same college!
- **State Level (SL) Seats**: Autonomous institutes (like COEP, VJTI, SPIT, Walchand, VIT Pune) fill 100% of their Maharashtra seats on State Level merit (no HU/OHU distinction).
        """
    },
    {
        "keywords": ["jee", "jee main", "all india", "non maharashtra", "ai quota"],
        "title": "How does All India (AI) Quota via JEE Main work?",
        "answer": """
**All India (AI) Quota Rules:**
- 15% seats in Maharashtra unaided private engineering colleges are reserved for All India candidates.
- **Primary Exam**: **JEE Main Paper-1 score** takes precedence over MHT-CET for All India seats.
- **Reservation**: All India seats are treated as General/Open merit (State caste reservations like OBC, SC, ST do NOT apply to AI quota seats).
- **Non-Maharashtra Students**: Can participate in CAP counseling solely based on JEE Main scores through this quota.
        """
    },
    {
        "keywords": ["round 2", "round 3", "spot round", "acap", "institutional round"],
        "title": "Round 2, Round 3, and Institutional (Spot) Rounds",
        "answer": """
**Counseling Round Strategies:**
- **Round 1 $\to$ Round 2**: Vacancies arise because students surrender lower choices or opt for Betterment. Cutoffs usually drop by 0.5% to 2.5%.
- **Round 3**: The final centralized online round. Usually has fewer vacancies in top CS/IT branches, but good core branch opportunities.
- **Institutional / Spot Rounds (ACAP)**: Conducted directly by individual colleges after CAP Round 3 for leftover vacant seats. Highly recommended for students with moderate percentiles wanting top tier institutes!
        """
    }
]

# College keyword mapping to 4-digit DTE code
COLLEGE_KEYWORDS = {
    "coep": ("6006", "COEP Technological University, Pune"),
    "vjti": ("3012", "VJTI Mumbai"),
    "spit": ("3215", "SPIT Mumbai"),
    "sardar patel": ("3215", "SPIT Mumbai"),
    "pict": ("6271", "PICT Pune"),
    "walchand": ("6007", "Walchand College of Engineering, Sangli"),
    "vit": ("6273", "VIT Pune"),
    "vishwakarma": ("6273", "VIT Pune"),
    "pccoe": ("6175", "PCCOE Pune"),
    "pimpri chinchwad": ("6175", "PCCOE Pune"),
    "dj sanghvi": ("3199", "D. J. Sanghvi College of Engineering, Mumbai"),
    "djsce": ("3199", "D. J. Sanghvi College of Engineering, Mumbai"),
    "thadomal": ("3182", "Thadomal Shahani Engineering College, Mumbai"),
    "somaiya": ("3014", "KJ Somaiya College of Engineering, Mumbai"),
    "cummins": ("6276", "Cummins College of Engineering for Women, Pune"),
    "mit alandi": ("6145", "MIT Academy of Engineering, Alandi"),
    "viit": ("6284", "VIIT Kondhwa, Pune"),
    "crce": ("3184", "Fr. CRCE Bandra, Mumbai"),
    "fr. agnel": ("3184", "Fr. CRCE Bandra, Mumbai"),
    "vesit": ("3185", "VESIT Chembur, Mumbai"),
    "vivekanand": ("3185", "VESIT Chembur, Mumbai"),
    "rcoem": ("4115", "RCOEM Nagpur"),
    "ramdeobaba": ("4115", "RCOEM Nagpur"),
    "gcoe amravati": ("1002", "Government College of Engineering, Amravati"),
    "gcoe karad": ("6007", "Government College of Engineering, Karad"),
    "gcoe aurangabad": ("2008", "Government College of Engineering, Aurangabad"),
    "aissms": ("6281", "AISSMS COE Pune"),
    "sinhgad": ("6178", "Sinhgad COE Vadgaon, Pune"),
    "dy patil akurdi": ("6272", "DY Patil College of Engineering, Akurdi"),
    "rait": ("3146", "RAIT Navi Mumbai"),
    "vidyalankar": ("3148", "Vidyalankar Institute of Technology, Mumbai"),
    "ycce": ("4123", "YCCE Nagpur")
}

class CounselingChatbot:
    def __init__(self, ml_engine=None):
        self.ml_engine = ml_engine

    def answer_query(self, user_text: str, candidate_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parses user query and determines whether it's:
        1. Greeting / introductory help
        2. College comparison ('COEP vs VJTI')
        3. College information inquiry ('Cutoff for PICT', 'SPIT placements')
        4. Recommendation query (explicit score or using active candidate profile)
        5. Counseling rules FAQ
        6. Informative fallback with prompt ideas
        """
        text_lower = str(user_text).lower().strip()

        # 1. Greetings & Casual Welcome
        if text_lower in ["hi", "hello", "hey", "hola", "namaste", "good morning", "good evening", "help"]:
            cand_name = candidate_profile.get("name", "Applicant") if candidate_profile else "Applicant"
            cand_score = candidate_profile.get("score_cet") or candidate_profile.get("score_jee") if candidate_profile else None
            score_text = f" (holding **{cand_score:.2f}%**)" if cand_score is not None else ""
            
            return {
                "type": "counseling_info",
                "title": f"Hello {cand_name}! How can I help you today?",
                "content": f"""
I am your **CounselAI Admissions Assistant**, actively aware of your profile{score_text}. Here are some things you can ask me:

- 🎯 *"Suggest colleges for my profile"* or *"What are my chances in Pune?"*
- 🏛️ *"What is the cutoff and placement package for PICT?"*
- ⚖️ *"Compare COEP and VJTI"* or *"PICT vs VIT Pune"*
- 🧊 *"Should I freeze or choose betterment in Round 1?"*
- 📜 *"What are the mandatory documents for OBC/EWS?"*
- 💰 *"What is the TFWS income limit and seat criteria?"*
                """.strip()
            }

        # 2. College Comparison Intent ('vs', 'compare', 'difference between')
        is_comparison = any(k in text_lower for k in [" vs ", " vs. ", "versus", "compare ", "difference between"])
        if is_comparison and self.ml_engine is not None:
            detected_colleges = []
            for kw, (code, full_name) in COLLEGE_KEYWORDS.items():
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    if (code, full_name) not in detected_colleges:
                        detected_colleges.append((code, full_name))
            
            if len(detected_colleges) >= 2:
                codes_to_comp = [c[0] for c in detected_colleges[:2]]
                comp_df = self.ml_engine.compare_colleges(codes_to_comp)
                if len(comp_df) > 0:
                    c1_name = detected_colleges[0][1]
                    c2_name = detected_colleges[1][1]
                    lines = [
                        f"### Side-by-Side Comparison: **{c1_name}** vs. **{c2_name}**\n",
                        "| Metric | " + " | ".join(comp_df["College Name"].tolist()) + " |",
                        "| :--- | " + " | ".join([":---"] * len(comp_df)) + " |"
                    ]
                    for col in ["Institutional Tier", "Average CTC", "Highest CTC", "Median CS/IT Cutoff", "Top Recruiters"]:
                        row_vals = [str(v) for v in comp_df[col].tolist()]
                        lines.append(f"| **{col}** | " + " | ".join(row_vals) + " |")
                    
                    lines.append("\n💡 *Counselor Recommendation: Both are premier institutions. Prioritize based on commute/location, autonomous curriculum, and specific branch specialization.*")
                    return {
                        "type": "recommendation",
                        "title": f"Comparison: {c1_name} vs. {c2_name}",
                        "content": "\n".join(lines)
                    }

        # 3. College-Specific Query (Cutoffs, Placements, Info)
        is_college_query = any(k in text_lower for k in ["cutoff", "placement", "package", "ctc", "salary", "recruiter", "about", "fees", "tier", "info"])
        for kw, (code, full_name) in COLLEGE_KEYWORDS.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                tier_info = COLLEGE_TIER_DATABASE.get(code, DEFAULT_TIER)
                
                # Fetch recent branch cutoffs from summary if available
                branch_cutoffs = []
                if self.ml_engine is not None and self.ml_engine.df_summary is not None:
                    c_rows = self.ml_engine.df_summary[self.ml_engine.df_summary["college_code"] == code]
                    if len(c_rows) > 0:
                        for b_cluster in ["Computer Engineering", "Information Technology", "AI & Data Science", "Electronics & Telecommunication", "Mechanical Engineering"]:
                            sub_b = c_rows[c_rows["branch_cluster"] == b_cluster]
                            if len(sub_b) > 0:
                                med_cut = round(float(sub_b["latest_cutoff"].median()), 2)
                                branch_cutoffs.append(f"• **{b_cluster}**: ~{med_cut}%")
                
                cutoff_text = "\n".join(branch_cutoffs[:4]) if branch_cutoffs else "• Recent cutoffs range from 88% to 99% across streams."

                return {
                    "type": "counseling_info",
                    "title": f"Institutional Overview: {full_name} (DTE `{code}`)",
                    "content": f"""
**🏛️ {full_name}**
- **Accreditation Tier**: {tier_info['tier']}
- **Average Placement CTC**: **{tier_info['avg_ctc']}** (Highest Package: **{tier_info['max_ctc']}**)
- **Key Campus Recruiters**: {tier_info['recruiters']}

**📊 Typical Closing Cutoffs (Open Category / Round 1):**
{cutoff_text}

*Tip: You can add {full_name} to your option form in the **📥 CAP Option Form** generator.*
                    """.strip()
                }

        # 4. College Recommendation Intent
        # A: Explicit Score in Text
        score_match = re.search(r"(\d{1,2}(?:\.\d{1,4})?)\s*(?:%|percentile|pct|marks)?", text_lower)
        is_recommendation_query = any(k in text_lower for k in [
            "suggest", "recommend", "colleges", "chances", "options", "predict", "get", 
            "eligible", "where can i get", "my chances", "best college", "what can i get"
        ])

        if score_match and is_recommendation_query and self.ml_engine is not None:
            score = float(score_match.group(1))
            if 10.0 <= score <= 100.0:
                return self._handle_recommendation_query(score, text_lower, candidate_profile)

        # B: Implicit Score from Active Candidate Profile
        if is_recommendation_query and candidate_profile is not None and self.ml_engine is not None:
            c_score = candidate_profile.get("score_cet") or candidate_profile.get("score_jee")
            if c_score and float(c_score) >= 10.0:
                return self._handle_recommendation_query(float(c_score), text_lower, candidate_profile)

        # 5. Check Counseling Rules Knowledge Base
        best_match = None
        highest_score = 0
        for entry in COUNSELING_KNOWLEDGE_BASE:
            matched_count = sum(1 for kw in entry["keywords"] if kw in text_lower)
            if matched_count > highest_score:
                highest_score = matched_count
                best_match = entry

        if best_match and highest_score > 0:
            return {
                "type": "counseling_info",
                "title": best_match["title"],
                "content": best_match["answer"].strip()
            }

        # 6. Default Fallback Guidance
        return {
            "type": "fallback",
            "title": "CounselAI Assistant Guide",
            "content": """
I didn't quite catch the specifics of your question. You can ask me:

- 🎯 *"Suggest colleges for my score"* or *"Recommend CS/IT colleges in Pune"*
- 🏛️ *"What are the placements and cutoffs for PICT or SPIT?"*
- ⚖️ *"Compare COEP vs VJTI"* or *"VIT vs PCCOE"*
- 🧊 *"Explain Self-Freeze vs Betterment"*
- 📜 *"What are the mandatory documents for CAP reporting?"*
- 💰 *"What is the TFWS fee waiver criteria and income limit?"*
            """.strip()
        }

    def _handle_recommendation_query(self, score: float, text: str, candidate_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handles conversational queries containing student percentile or using candidate profile."""
        chosen_cities = []
        for city_key, city_canonical in [
            ("pune", "Pune"), ("mumbai", "Mumbai / MMR"), ("thane", "Mumbai / MMR"), 
            ("navi mumbai", "Mumbai / MMR"), ("nagpur", "Nagpur"), ("nashik", "Nashik"),
            ("aurangabad", "Chhatrapati Sambhaji Nagar"), ("sambhaji", "Chhatrapati Sambhaji Nagar")
        ]:
            if city_key in text and city_canonical not in chosen_cities:
                chosen_cities.append(city_canonical)

        if not chosen_cities and candidate_profile and candidate_profile.get("cities"):
            chosen_cities = candidate_profile.get("cities")

        branches = []
        if any(b in text for b in ["cs", "computer", "cse"]):
            branches.append("Computer Engineering")
        if any(b in text for b in ["it", "information technology"]):
            branches.append("Information Technology")
        if any(b in text for b in ["ai", "ds", "data science", "aiml"]):
            branches.append("AI & Data Science")
        if any(b in text for b in ["entc", "e&tc", "telecom", "electronics"]):
            branches.append("Electronics & Telecommunication")
        if any(b in text for b in ["mech", "mechanical"]):
            branches.append("Mechanical Engineering")
        if any(b in text for b in ["civil"]):
            branches.append("Civil Engineering")

        if not branches and candidate_profile and candidate_profile.get("branches"):
            branches = candidate_profile.get("branches")

        exam_mode = "MHT-CET (MH Candidates)"
        if "jee" in text and "cet" not in text:
            exam_mode = "JEE Main Only (All India / Non-MH)"
        elif candidate_profile and candidate_profile.get("exam_mode"):
            exam_mode = candidate_profile.get("exam_mode")

        category = candidate_profile.get("category", "OPEN") if candidate_profile else "OPEN"
        gender = candidate_profile.get("gender", "General") if candidate_profile else "General"

        preds = self.ml_engine.predict_choices(
            score_cet=score if "jee" not in text else None,
            score_jee=score if "jee" in text else None,
            exam_mode=exam_mode,
            category=category,
            gender=gender,
            selected_cities=chosen_cities if chosen_cities else None,
            selected_branches=branches if branches else None
        )

        all_df = preds.get("all_ordered")
        if all_df is None or len(all_df) == 0:
            return {
                "type": "counseling_info",
                "title": f"Recommendations for {score:.1f}%",
                "content": f"No colleges directly matched your specific filters for **{score:.1f}%**. Try broadening your city or branch selection in your Candidate Profile!"
            }

        top_recs = all_df.head(6)
        response_lines = [
            f"Here are the top strategic choices evaluated for **{score:.2f}%** ({', '.join(chosen_cities) if chosen_cities else 'Preferred Cities'} • {category}):\n",
            "| Preference | College | Branch | Placement | Suitability |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ]

        for _, r in top_recs.iterrows():
            badge_clean = r['category_tag']
            c_disp = r['college_name'][:38] + ("..." if len(r['college_name']) > 38 else "")
            b_disp = r['branch'][:24] + ("..." if len(r['branch']) > 24 else "")
            response_lines.append(f"| **{badge_clean}** | {c_disp} | {b_disp} | {r['avg_placement']} | **{r['suitability_pct']}%** |")

        response_lines.append("\n💡 *Tip: Go to **🎯 Admission Predictor** for your complete sorted sequence or **📥 CAP Option Form** to export your official choice codes!*")

        return {
            "type": "recommendation",
            "title": f"Strategic Analysis for {score:.2f}% Percentile",
            "content": "\n".join(response_lines)
        }

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from ml_engine import AdmissionMLEngine
    bot = CounselingChatbot(AdmissionMLEngine())
    print("1. Testing FAQ answer:")
    res = bot.answer_query("What happens if I choose betterment in round 1?")
    print(res["title"])
    print("\n2. Testing college info lookup:")
    info = bot.answer_query("What is the cutoff for COEP?")
    print(info["title"])
    print(info["content"][:200] + "...")
    print("\n3. Testing comparison:")
    comp = bot.answer_query("Compare COEP vs VJTI")
    print(comp["title"])
    print(comp["content"][:200] + "...")
    print("\n4. Testing candidate-aware suggestion:")
    rec = bot.answer_query("suggest colleges for me", {"name": "Priya", "score_cet": 93.4, "category": "OPEN", "cities": ["Pune"]})
    print(rec["title"])
    print(rec["content"][:200] + "...")
