"""
ML Engine & Recommendation Core for MHT-CET / JEE Main Admissions.
Includes:
- Gradient-Boosted Cutoff Prediction (HistGradientBoostingRegressor)
- Suitability & Probability Scoring (Normal CDF / Quantile Distance)
- Strict Ambitious -> Target -> Safe Categorization
- Round 2/3 Betterment Upgrade Simulator
- College Comparison Engine
- ML Performance Metrics for College Viva Review
"""

import os
import math
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from data_processor import (
    load_and_preprocess,
    PROCESSED_RAW_FILE,
    PROCESSED_SUMMARY_FILE,
    CACHE_DIR,
    COLLEGE_TIER_DATABASE,
    DEFAULT_TIER
)

MODEL_FILE = os.path.join(CACHE_DIR, "cutoff_model.joblib")
METRICS_FILE = os.path.join(CACHE_DIR, "model_metrics.joblib")

def normal_cdf(x):
    """Cumulative distribution function for standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

class AdmissionMLEngine:
    def __init__(self):
        self.model = None
        self.metrics = {}
        self.df_summary = None
        self.df_raw = None
        self.load_data()
        self.load_or_train_model()

    def load_data(self):
        """Loads summary and raw datasets."""
        self.df_summary = load_and_preprocess(force_reload=False)
        if os.path.exists(PROCESSED_RAW_FILE):
            self.df_raw = pd.read_parquet(PROCESSED_RAW_FILE)

    def load_or_train_model(self, force_retrain=False):
        """Loads cached model or trains a HistGradientBoostingRegressor on 337k+ rows."""
        if not force_retrain and os.path.exists(MODEL_FILE) and os.path.exists(METRICS_FILE):
            try:
                self.model = joblib.load(MODEL_FILE)
                self.metrics = joblib.load(METRICS_FILE)
                print(f"Loaded trained ML model (R2: {self.metrics.get('r2', 'N/A'):.4f}, MAE: {self.metrics.get('mae', 'N/A'):.2f}%)")
                return
            except Exception as e:
                print(f"Error loading model: {e}, retraining...")

        print("Training HistGradientBoostingRegressor on historical cutoff records...")
        if self.df_raw is None or len(self.df_raw) == 0:
            self.df_raw = pd.read_parquet(PROCESSED_RAW_FILE)

        # Feature preparation
        # College code has cardinality 466 > 255, so we use Target Encoding (historical mean cutoff per college)
        college_means = self.df_raw.groupby("college_code")["percentile"].mean().to_dict()
        df_train = self.df_raw.dropna(subset=["percentile"]).copy()
        df_train["college_enc"] = df_train["college_code"].map(college_means).fillna(50.0)

        # Categorical features with cardinality < 255
        cat_cols = ["branch_cluster", "quota", "category", "gender", "metro_city"]
        for col in cat_cols:
            df_train[col] = df_train[col].astype("category").cat.codes

        features = ["year", "round", "avg_lpa", "college_enc"] + cat_cols
        X = df_train[features]
        y = df_train["percentile"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        # Train model with early stopping
        model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=45,
            min_samples_leaf=25,
            random_state=42,
            categorical_features=[features.index(c) for c in cat_cols]
        )
        model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        self.model = model
        self.metrics = {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "algorithm": "Histogram Gradient-Boosted Decision Trees (LightGBM/HistGradientBoosting)",
            "features_used": features
        }

        joblib.dump(self.model, MODEL_FILE)
        joblib.dump(self.metrics, METRICS_FILE)
        joblib.dump(college_means, os.path.join(CACHE_DIR, "college_enc.joblib"))
        print(f"Model trained successfully! Test R2: {r2:.4f}, MAE: {mae:.2f}%")

    def predict_choices(self, 
                        score_cet=None, 
                        score_jee=None, 
                        exam_mode="MHT-CET (MH Candidates)", 
                        category="OPEN", 
                        gender="General", 
                        selected_cities=None, 
                        selected_branches=None, 
                        round_target="Round 1",
                        min_suitability=15):
        """
        Predicts and classifies college choices into Ambitious, Target, and Safe.
        Respects exam choice and prioritizes MH quota if 'Both' is selected.
        """
        df = self.df_summary.copy()

        # 1. Determine Effective Score & Quota Filter
        if exam_mode == "JEE Main Only (All India / Non-MH)":
            active_score = float(score_jee) if score_jee is not None else 0.0
            # JEE Main candidates only compete for All India (AI) seats
            df = df[df["quota"] == "AI"]
        elif exam_mode == "MHT-CET (MH Candidates)":
            active_score = float(score_cet) if score_cet is not None else 0.0
            # MH candidates compete primarily in MH quota
            df = df[df["quota"] == "MH"]
        else:  # "Both (Prioritize MH Quota)"
            # Prioritize MH quota, using MHT-CET score; allow AI quota if JEE score is competitive
            cet_val = float(score_cet) if score_cet is not None else 0.0
            jee_val = float(score_jee) if score_jee is not None else 0.0
            active_score = max(cet_val, jee_val)
            # Default to MH unless JEE is significantly higher
            if jee_val > cet_val + 5.0:
                df = df[df["quota"].isin(["MH", "AI"])]
            else:
                # Prefer MH quota
                df = df[df["quota"] == "MH"]

        # 2. Category & Gender Filter
        # If AI quota, category is typically OPEN/General
        if len(df[df["category"] == category]) > 0:
            df = df[df["category"] == category]
        else:
            df = df[df["category"] == "OPEN"]

        if gender == "Ladies":
            # Ladies can take both Ladies and General seats
            df = df[df["gender"].isin(["Ladies", "General"])]
        else:
            df = df[df["gender"] == "General"]

        # 3. City Filter
        if selected_cities and len(selected_cities) > 0 and "All Cities" not in selected_cities:
            df = df[df["metro_city"].isin(selected_cities)]

        # 4. Branch Filter
        if selected_branches and len(selected_branches) > 0 and "All Branches" not in selected_branches:
            df = df[df["branch_cluster"].isin(selected_branches)]

        if len(df) == 0:
            return {"ambitious": [], "target": [], "safe": [], "all_ordered": pd.DataFrame()}

        # 5. Round Cutoff Target Selection
        if round_target == "Round 2":
            cutoff_col = "r2_cutoff"
        elif round_target == "Round 3":
            cutoff_col = "r3_cutoff"
        elif round_target == "Best of All Rounds":
            cutoff_col = "min_cutoff"
        else:
            cutoff_col = "r1_cutoff"

        # 6. Compute Predicted Cutoff & Suitability / Probability
        results = []
        for _, row in df.iterrows():
            base_cutoff = row[cutoff_col]
            drift = row["cutoff_drift"]
            std_dev = max(row["cutoff_std"], 0.6)

            # Projected next-cycle cutoff with drift adjustment
            predicted_cutoff = base_cutoff + (0.5 * drift)
            predicted_cutoff = max(0.0, min(100.0, predicted_cutoff))

            diff = active_score - predicted_cutoff

            # Probability via Standard Normal CDF
            z = diff / std_dev
            prob = normal_cdf(z)
            suitability_pct = int(round(prob * 100))
            suitability_pct = max(1, min(99, suitability_pct))

            # Categorization Logic
            if diff >= 0.7 or prob >= 0.78:
                cat_tag = "Safe"
                sort_priority = 3
                badge = "🛡️ Safe"
            elif -1.5 <= diff < 0.7 or 0.38 <= prob < 0.78:
                cat_tag = "Target"
                sort_priority = 2
                badge = "🎯 Target"
            elif -4.0 <= diff < -1.5 or 0.12 <= prob < 0.38:
                cat_tag = "Ambitious"
                sort_priority = 1
                badge = "🔥 Ambitious"
            else:
                continue  # Exclude extreme unreachables (< 12% probability)

            if suitability_pct < min_suitability:
                continue

            results.append({
                "college_code": row["college_code"],
                "college_name": row["clean_college_name"],
                "branch": row["course_name"],
                "branch_cluster": row["branch_cluster"],
                "city": row["metro_city"],
                "tier": row["college_tier"],
                "avg_placement": row["avg_placement_ctc"],
                "avg_lpa": row["avg_lpa"],
                "max_placement": row["max_placement_ctc"],
                "top_recruiters": row["top_recruiters"],
                "historical_cutoff": round(base_cutoff, 2),
                "predicted_cutoff": round(predicted_cutoff, 2),
                "percentile_diff": round(diff, 2),
                "suitability_pct": suitability_pct,
                "category_tag": cat_tag,
                "badge": badge,
                "sort_priority": sort_priority,
                "choice_code": f"{row['college_code']}{abs(hash(row['course_name'])) % 1000:03d}10"  # DTE style Choice Code
            })

        df_res = pd.DataFrame(results)
        if len(df_res) == 0:
            return {"ambitious": [], "target": [], "safe": [], "all_ordered": pd.DataFrame()}

        # Strict preference ordering: Ambitious (Priority 1) -> Target (Priority 2) -> Safe (Priority 3)
        # Within each category, sorted by higher suitability and higher placement package
        df_res = df_res.sort_values(
            by=["sort_priority", "suitability_pct", "avg_lpa"],
            ascending=[True, False, False]
        ).reset_index(drop=True)

        ambitious_list = df_res[df_res["category_tag"] == "Ambitious"].to_dict(orient="records")
        target_list = df_res[df_res["category_tag"] == "Target"].to_dict(orient="records")
        safe_list = df_res[df_res["category_tag"] == "Safe"].to_dict(orient="records")

        return {
            "ambitious": ambitious_list,
            "target": target_list,
            "safe": safe_list,
            "all_ordered": df_res
        }

    def simulate_betterment(self, allotted_college_code, allotted_branch, student_score, category="OPEN", selected_cities=None):
        """
        Betterment Upgrade Simulator:
        Analyzes Round 1 -> Round 2 and Round 3 cutoff drops across higher-tier colleges.
        Identifies realistic upgrade opportunities for a student choosing Betterment (Float).
        """
        df = self.df_summary.copy()
        df = df[df["category"] == category]

        if selected_cities and len(selected_cities) > 0 and "All Cities" not in selected_cities:
            df = df[df["metro_city"].isin(selected_cities)]

        # Find current allotted college stats
        current_rows = df[(df["college_code"] == allotted_college_code) & (df["course_name"] == allotted_branch)]
        current_r1_cutoff = current_rows["r1_cutoff"].iloc[0] if len(current_rows) > 0 else student_score

        # Filter for colleges that are higher preference (higher Round 1 cutoff)
        higher_df = df[df["r1_cutoff"] > current_r1_cutoff].copy()

        upgrade_candidates = []
        for _, row in higher_df.iterrows():
            r1 = row["r1_cutoff"]
            r2 = row["r2_cutoff"]
            r3 = row["r3_cutoff"]

            # Historical drop between Round 1 and subsequent rounds
            r2_drop = r1 - r2
            r3_drop = r1 - r3

            # Check if student's score enters the Round 2 or Round 3 window
            score_diff_r2 = student_score - r2
            score_diff_r3 = student_score - r3

            std_dev = max(row["cutoff_std"], 0.6)
            prob_r2 = normal_cdf(score_diff_r2 / std_dev)
            prob_r3 = normal_cdf(score_diff_r3 / std_dev)

            best_prob = max(prob_r2, prob_r3)
            upgrade_chance = int(round(best_prob * 100))

            # Only recommend realistic upgrades (between 25% and 85% upgrade chance)
            if 25 <= upgrade_chance <= 90:
                upgrade_candidates.append({
                    "college_code": row["college_code"],
                    "college_name": row["clean_college_name"],
                    "branch": row["course_name"],
                    "city": row["metro_city"],
                    "tier": row["college_tier"],
                    "avg_placement": row["avg_placement_ctc"],
                    "r1_cutoff": round(r1, 2),
                    "r2_cutoff": round(r2, 2),
                    "r3_cutoff": round(r3, 2),
                    "avg_drop": round(max(r2_drop, r3_drop), 2),
                    "upgrade_chance_pct": upgrade_chance,
                    "target_round": "Round 2" if prob_r2 >= prob_r3 else "Round 3"
                })

        df_upgrades = pd.DataFrame(upgrade_candidates)
        if len(df_upgrades) > 0:
            df_upgrades = df_upgrades.sort_values(by="upgrade_chance_pct", ascending=False).reset_index(drop=True)
        return df_upgrades

    def compare_colleges(self, college_codes):
        """
        Side-by-side comparison matrix for 2 or 3 selected colleges.
        """
        if not college_codes:
            return pd.DataFrame()

        records = []
        for code in college_codes:
            sub = self.df_summary[self.df_summary["college_code"] == code]
            if len(sub) == 0:
                continue
            first_row = sub.iloc[0]
            tier_info = COLLEGE_TIER_DATABASE.get(code, DEFAULT_TIER)
            branches = sorted(sub["branch_cluster"].unique().tolist())
            
            cs_it_cutoffs = sub[sub["branch_cluster"].isin(["Computer Engineering", "Information Technology", "AI & Data Science"])]
            median_cs_cutoff = round(float(cs_it_cutoffs["latest_cutoff"].median()), 2) if len(cs_it_cutoffs) > 0 else "N/A"
            
            records.append({
                "College Code": code,
                "College Name": first_row["clean_college_name"],
                "City": first_row["metro_city"],
                "Institutional Tier": tier_info["tier"],
                "Average CTC": tier_info["avg_ctc"],
                "Highest CTC": tier_info["max_ctc"],
                "Top Recruiters": tier_info["recruiters"],
                "Median CS/IT Cutoff": median_cs_cutoff,
                "Key Branches Offered": ", ".join(branches[:4]) + ("..." if len(branches) > 4 else "")
            })

        return pd.DataFrame(records)

    def get_all_cities(self):
        """Returns unique list of normalized metro cities."""
        return sorted(self.df_summary["metro_city"].unique().tolist())

    def get_all_branches(self):
        """Returns unique list of canonical branch clusters."""
        return sorted(self.df_summary["branch_cluster"].unique().tolist())

    def get_all_colleges(self):
        """Returns dictionary of college_code -> clean_college_name."""
        df_colleges = self.df_summary[["college_code", "clean_college_name"]].drop_duplicates()
        return dict(zip(df_colleges["college_code"], df_colleges["clean_college_name"]))

if __name__ == "__main__":
    engine = AdmissionMLEngine()
    print("Testing ML Engine prediction with MHT-CET 93.5% in Pune...")
    preds = engine.predict_choices(
        score_cet=93.5, 
        exam_mode="MHT-CET (MH Candidates)", 
        selected_cities=["Pune"], 
        selected_branches=["Computer Engineering", "Information Technology"]
    )
    print("Ambitious count:", len(preds['ambitious']))
    print("Target count:", len(preds['target']))
    print("Safe count:", len(preds['safe']))
    if len(preds['all_ordered']) > 0:
        print("Top 3 choices preview:")
        for idx, row in preds['all_ordered'].head(3).iterrows():
            print(f"[{row['category_tag']}] {row['college_name']} - {row['branch']} | Suitability: {row['suitability_pct']}% | Cutoff: {row['predicted_cutoff']}%")
