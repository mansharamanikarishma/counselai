"""
Smart Offline AI Counseling Assistant for MHT-CET & JEE Main Candidates.
Dual-Functionality:
1. Answers CAP counseling rules, document requirements, TFWS, and Betterment/Freeze questions.
2. Natural language college query intent matching (e.g., 'I scored 94 in CET, suggest IT colleges in Pune').
Zero external API tokens consumed — 100% local rule & semantic intent matching.
"""

import re
from typing import Dict, Any, List

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

class CounselingChatbot:
    def __init__(self, ml_engine=None):
        self.ml_engine = ml_engine

    def answer_query(self, user_text: str) -> Dict[str, Any]:
        """
        Parses user query and determines whether it's a rule query or a score/recommendation query.
        """
        text_lower = user_text.lower().strip()

        # 1. Check if the user is asking for college recommendations with marks/percentile
        score_match = re.search(r"(\d{1,2}(?:\.\d{1,4})?)\s*(?:%|percentile|pct|marks)?", text_lower)
        is_recommendation_query = any(k in text_lower for k in ["suggest", "recommend", "colleges", "chances", "options", "predict", "get"])

        if score_match and is_recommendation_query and self.ml_engine is not None:
            score = float(score_match.group(1))
            if 10.0 <= score <= 100.0:
                return self._handle_recommendation_query(score, text_lower)

        # 2. Check counseling rules knowledge base
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

        # 3. Default fallback guidance
        return {
            "type": "fallback",
            "title": "How I can assist you:",
            "content": """
I am your **Maharashtra CAP Counseling Assistant**. You can ask me questions like:
- *"What is Betterment and should I take it?"*
- *"When should I choose Self-Freeze?"*
- *"What are the mandatory documents for OBC/EWS?"*
- *"What is TFWS and what is the income limit?"*
- *"How does All India Quota work via JEE Main?"*
- *"I scored 94 percentile in MHT-CET, suggest Computer Engineering in Pune"*
            """
        }

    def _handle_recommendation_query(self, score: float, text: str) -> Dict[str, Any]:
        """Handles conversational queries containing student percentile."""
        # Detect city
        cities = ["Pune", "Mumbai / MMR", "Nagpur", "Nashik", "Chhatrapati Sambhaji Nagar"]
        chosen_cities = []
        if "pune" in text:
            chosen_cities.append("Pune")
        if any(m in text for m in ["mumbai", "thane", "navi mumbai"]):
            chosen_cities.append("Mumbai / MMR")
        if "nagpur" in text:
            chosen_cities.append("Nagpur")
        if "nashik" in text:
            chosen_cities.append("Nashik")

        # Detect branch
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

        # Detect exam
        exam_mode = "JEE Main Only (All India / Non-MH)" if "jee" in text and "cet" not in text else "MHT-CET (MH Candidates)"

        preds = self.ml_engine.predict_choices(
            score_cet=score if "jee" not in text else None,
            score_jee=score if "jee" in text else None,
            exam_mode=exam_mode,
            selected_cities=chosen_cities if chosen_cities else None,
            selected_branches=branches if branches else None
        )

        all_df = preds.get("all_ordered")
        if all_df is None or len(all_df) == 0:
            return {
                "type": "counseling_info",
                "title": f"Recommendations for {score}%",
                "content": f"No colleges directly matched your specific filters for {score}%. Try broadening your city or branch selection in the sidebar!"
            }

        # Build clean markdown list
        top_recs = all_df.head(6)
        response_lines = [
            f"Here are the top strategic choices for **{score}%** ({', '.join(chosen_cities) if chosen_cities else 'All Cities'}):",
            "",
            "| Preference | College | Branch | Placement | Suitability |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ]

        for _, r in top_recs.iterrows():
            badge_clean = r['category_tag']
            response_lines.append(f"| **{badge_clean}** | {r['college_name'][:38]} | {r['branch'][:25]} | {r['avg_placement']} | **{r['suitability_pct']}%** |")

        response_lines.append("\n*Tip: Go to **Tab 1 (Admission Predictor)** to customize your full preference list and download your official CAP choice form!*")

        return {
            "type": "recommendation",
            "title": f"Strategic Analysis for {score}% Percentile",
            "content": "\n".join(response_lines)
        }

if __name__ == "__main__":
    from ml_engine import AdmissionMLEngine
    bot = CounselingChatbot(AdmissionMLEngine())
    print("Testing FAQ answer:")
    res = bot.answer_query("What happens if I choose betterment in round 1?")
    print(res["title"])
    print(res["content"][:180] + "...")
    print("\nTesting conversational recommendation:")
    rec = bot.answer_query("I got 93 percentile, recommend CS in Pune")
    print(rec["title"])
    print(rec["content"][:250] + "...")
