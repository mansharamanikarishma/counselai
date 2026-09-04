# CounselAI: Decision Support System for Engineering Admissions

**A College Minor Project built with Python, Scikit-Learn, and Streamlit.**

**CounselAI** is an intelligent decision support system analyzing 5 years of historical Maharashtra Centralized Admission Process (CAP) engineering counseling data (**337,776 valid records**) to predict and recommend **Ambitious**, **Target**, and **Safe** college options with individual suitability scores.

---

## 🌟 Key Features

1. **🎯 Personalized Admission Predictor (Idea B)**:
   - Evaluates scores across **MHT-CET** (Maharashtra Quota) and **JEE Main** (All India Quota).
   - Generates 3 distinct categorized views: **Ambitious** (15–40% dream choices), **Target** (40–80% realistic matches), and **Safe** (80%+ guaranteed safety nets).
   - Computes individual calibrated **Suitability / Admission Probability Scores (0–100%)**.

2. **📥 1-Click Official CAP Option Form Generator**:
   - Strictly sequences choices as **Ambitious $\to$ Target $\to$ Safe** to prevent premature auto-allotments.
   - Attaches official DTE Choice Codes and allows instant **CSV export** formatted for direct reference during official CAP portal choice filling.

3. **🏛️ Side-by-Side College Comparison Matrix**:
   - Compare 2 to 4 colleges simultaneously across average placement CTC, highest CTC, institutional tier, median cutoffs, top recruiters, and branches.

4. **🔄 Round 2/3 Betterment Upgrade Simulator (Idea A)**:
   - Answers: *"If I got College X in Round 1, what higher colleges can I realistically upgrade to in Round 2 or 3?"*
   - Analyzes historical multi-round cutoff drops ($\Delta R_1 \to R_2 \to R_3$) to calculate realistic upgrade probabilities.

5. **⚖️ Freeze vs. Betterment Candidate Strategy Guide**:
   - Clear candidate action guide comparing **Self-Freeze** vs. **Betterment (Float)**, explaining the ₹1,000 Seat Acceptance Fee and seat retention rules. Strictly excludes Auto-Freeze.

6. **💼 Placement & Institutional Directory**:
   - Enriched institutional tier database (COEP, VJTI, SPIT, PICT, VIT Pune, PCCOE, DJ Sanghvi, etc.) with approximate average and highest CTC figures.

7. **📈 5-Year Interactive Cutoff Trajectory Explorer**:
   - Plotly time-series charts displaying year-over-year branch cutoff trends (2020–2024).

8. **🤖 Smart Offline Counseling Chatbot**:
   - Local rule and natural language intent-matching engine answering counseling rules, TFWS eligibility, mandatory documents, and conversational recommendations without consuming external API credits.

9. **📊 ML Model Insights & Academic Viva Dashboard**:
   - Explains model metrics ($R^2$, MAE, RMSE), training on 337k+ rows, and the statistical formulation of $\Phi(z)$ probability calibration.

---

## 🏗️ Project Architecture

```
mhtcet_predictor/
│
├── cache/                               # Auto-generated high-speed cached artifacts
│   ├── cleaned_cap_data.parquet         # 337,776 clean historical rows
│   ├── processed_summary.parquet        # 40,533 unique aggregated option records
│   ├── cutoff_model.joblib              # Trained HistGradientBoostingRegressor
│   └── model_metrics.joblib             # Test set MAE, RMSE, and R2 evaluation metrics
│
├── data_processor.py                    # Data cleaning, city/branch normalization & placement enrichment
├── ml_engine.py                         # ML training, suitability scoring, betterment simulator
├── counselor_bot.py                     # Offline counseling FAQ & conversational intent matcher
├── betterment_guide.py                  # Self-Freeze vs Betterment educational guide
├── app.py                               # Master Streamlit 9-tab web application
├── requirements.txt                     # Project dependencies
└── README.md                            # Comprehensive documentation & viva guide
```

---

## 🚀 How to Run Locally

1. Open your terminal in this directory:
   ```bash
   cd "C:\Users\Karishma\.gemini\antigravity\scratch\mhtcet_predictor"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit portal:
   ```bash
   streamlit run app.py
   ```
   The application will automatically open in your web browser at `http://localhost:8501`!

---

## 🌐 How to Deploy for Free on Streamlit Community Cloud

You can host this application permanently with a free public URL (e.g. `https://mhtcet-counselor.streamlit.app`) in 3 minutes:

1. **Initialize a Git repository and push to GitHub**:
   ```bash
   git init
   git add app.py ml_engine.py data_processor.py counselor_bot.py betterment_guide.py requirements.txt README.md cache/
   git commit -m "Initial release of MHT-CET Admission Predictor & Counseling Portal"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/mhtcet-predictor.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
   - Click **"New app"**.
   - Select your repository (`mhtcet-predictor`), branch (`main`), and set Main file path to `app.py`.
   - Click **"Deploy!"**.
   - Your live public web app is ready and accessible to anyone on the internet for free!

---

## 🎓 Viva / Mentor Defense Q&A Sheet

**Q1: Why did you choose a gradient-boosted tree ensemble over basic linear regression?**
> *Answer*: Cutoffs in CAP counseling exhibit non-linear interactions across categories, autonomous status, and counseling rounds. Tree-based gradient boosting (`HistGradientBoostingRegressor`) natively bins continuous features and captures non-linear category-tier interactions without assuming normality of features.

**Q2: How does the system calculate Admission Probability & Suitability?**
> *Answer*: Rather than a rigid yes/no binary cutoff, we calculate the student's percentile distance $\Delta = S - \hat{C}$ relative to the branch's historical standard deviation $\sigma$. The probability is derived using the cumulative normal distribution $\Phi\left(\frac{\Delta}{\sigma}\right)$.

**Q3: Why must Ambitious choices appear first in the preference order?**
> *Answer*: In Maharashtra CAP counseling, if an applicant places a safe college as Choice #1, they are locked into that college upon allotment. By sequencing **Ambitious $\to$ Target $\to$ Safe**, students maximize their opportunity to be considered for dream colleges in early rounds while retaining target and safe colleges as a guaranteed safety net.
