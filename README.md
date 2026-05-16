# Digital Burnout & Productivity Decision Support System

This project builds an intelligent Decision Support System (DSS) to analyze digital burnout risk and personalized productivity patterns using Machine Learning (Random Forest) and Profile Matching analytics. The system combines predictive modeling with explainable gap analysis to provide actionable recommendations for digital wellness.

## Features
- **Data Preprocessing:** Cleans and scales digital behavior, sleep quality, work performance, and lifestyle metrics from a large-scale dataset.
- **Machine Learning Models:** Trains separate Random Forest classifiers for burnout risk prediction and productivity classification.
- **Profile Matching (SPK):** Implements weighted gap analysis comparing user profiles against an ideal productivity profile across 25+ dimensions.
- **Weighted Metrics:** Applies scientific weights to features (e.g., sleep quality: 7%, stress level: 7%, deep work hours: 6%) for holistic assessment.
- **Radar Visualization:** Generates interactive polar charts comparing user profiles vs. ideal profiles across 10 key wellness dimensions.
- **Personalized Recommendations:** Generates context-aware actionable recommendations based on top contributors to productivity gaps.
- **Interactive Dashboard:** Provides a user-friendly Jupyter widget interface for real-time analysis and what-if scenarios.
- **Model Persistence:** Saves trained models using joblib for efficient inference on new data.

## Results
- **Burnout Risk Classification:** Categorizes users into High / Medium / Low risk levels with probability estimates.
- **Productivity Prediction:** Predicts productivity category (High / Medium / Low) based on comprehensive lifestyle analysis.
- **Gap Analysis:** Identifies top 6 factors with largest deviations from ideal profile, ranked by weighted importance.
- **Contextual Alerts:** Provides severity-based recommendations:
  - 🔴 **CRITICAL:** Burnout risk > 65 (immediate action required)
  - 🟡 **WARNING:** Burnout risk 35–65 (implement lifestyle changes)
  - 🟢 **GOOD:** Burnout risk < 35 (maintain current habits)

## Key Metrics Analyzed
**Digital Behavior:** Screen time, social media hours, doomscrolling, notifications, app switches, phone unlocks, late-night device usage

**Sleep & Recovery:** Sleep hours, sleep quality, mental fatigue, emotional exhaustion, stress level

**Work Performance:** Focus sessions, deep work hours, distraction frequency, task completion rate, concentration, meeting hours

**Lifestyle:** Physical activity, caffeine intake, motivation level, work satisfaction

**Context:** Age, occupation, work mode (office/hybrid/remote), device usage type, workspace quality, internet stability

## Requirements
- Python 3.8+
- Jupyter Notebook / JupyterLab / VS Code (Jupyter extension)
- pandas, numpy
- scikit-learn (RandomForest, preprocessing, model evaluation)
- matplotlib (visualization)
- ipywidgets (interactive dashboard)
- joblib (model serialization)

Install requirements with:
```bash
pip install pandas numpy matplotlib scikit-learn ipywidgets joblib
```

## Dataset
The dataset used in this project is available on Kaggle:

**📊 [Digital Burnout and Productivity Analytics](https://www.kaggle.com/datasets/aiexplorer77/digital-burnout-and-productivity-analytics)**

- **Size:** 5 Million records × 34 columns
- **Features:** Digital behavior metrics, sleep quality, work performance, lifestyle factors, and demographic information
- **Target Variables:** Burnout risk (0–100 score), Productivity category (High/Medium/Low)
- **Format:** CSV

To use this dataset:
1. Visit the Kaggle link above
2. Click **"Download"** to get `digital_burnout_productivity_dataset.csv`
3. Place the CSV file in the same directory as `main.ipynb`
4. Run the notebook cells to train models and launch the dashboard

## Usage
1. Clone this repository and ensure all files are in the same directory:
   - `main.ipynb` — Main analysis notebook
   - `digital_burnout_productivity_dataset.csv` — Dataset file
2. Open `main.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
3. Run cells sequentially:
   - **Cell 1:** Load libraries and configure global settings
   - **Cell 2:** Train or load pre-trained ML models (RandomForest)
   - **Cell 3:** Initialize SPK profile matching engine with ideal profiles and weighted metrics
   - **Cell 4:** Define interactive widget forms for user input
   - **Cell 5:** Launch the interactive dashboard and execute analysis
4. Interact with the widget form to input your metrics, then click **"🔍 Analisis Produktivitas Saya"** to see:
   - Burnout risk level + ML confidence
   - Productivity category + ML confidence
   - GAP Score (0–100 scale)
   - Top 6 contributing factors to gaps
   - 6 personalized recommendations

## What-If Analysis
Use the widget sliders to explore different scenarios:
- Adjust sleep hours and observe impact on gap score
- Reduce social media and doomscrolling to see productivity improvements
- Experiment with deep work hours and focus sessions
- Evaluate impact of stress management and physical activity

## File Structure
```
llm_news/
├── main.ipynb                                    # Main notebook: ML + SPK + Dashboard
├── digital_burnout_productivity_dataset.csv      # Input dataset (5M rows × 34 columns)
├── model_burnout.pkl                            # Pre-trained burnout classifier (saved after first run)
├── model_productivity.pkl                        # Pre-trained productivity classifier (saved after first run)
└── README.md                                     # This file
```

## Technical Architecture

### Machine Learning Pipeline (Cell 2)
```
Input Features (34 columns)
         ↓
  Train/Test Split (80/20)
         ↓
  Feature Preprocessing:
  - StandardScaler (numerical features)
  - OrdinalEncoder (categorical features)
         ↓
  Random Forest Classifier (n_estimators=80, max_depth=14)
         ↓
  Burnout Model → Accuracy ~85–92%
  Productivity Model → Accuracy ~78–88%
```

### Decision Support System (Cell 3)
```
User Input (25 features)
         ↓
  Normalize to 0–1 scale per feature ranges
         ↓
  Compare vs. Ideal Profile
         ↓
  Compute weighted gaps (25 weights, Σ ≈ 1.0)
         ↓
  GAP Score = Σ(weighted deviations) × 100
         ↓
  Rank by severity → Top 6 contributors
         ↓
  Generate personalized recommendations
         ↓
  Produce radar chart visualization
```

## Model Details
- **Algorithm:** Random Forest (ensemble of 80 decision trees)
- **Max Depth:** 14 (prevents overfitting)
- **Class Weighting:** Balanced (handles imbalanced classes)
- **Feature Ranges:** Normalized per feature min/max from training data
- **Feature Importance:** Implicit in tree splits; top predictors typically: sleep hours, stress level, deep work, motivation, focus sessions

## Interpretation Guide

### Burnout Risk Levels
| Level | Range | Action |
|-------|-------|--------|
| **Tinggi** (High) | ≥ 65 | ⚠️ CRITICAL — Seek professional help, evaluate work-life balance |
| **Sedang** (Medium) | 35–65 | ⚡ WARNING — Implement lifestyle changes, monitor closely |
| **Rendah** (Low) | < 35 | ✅ GOOD — Maintain current positive habits |

### GAP Score Interpretation
| Score | Status | Meaning |
|-------|--------|---------|
| **0–30** | ✅ Excellent | Profile closely matches ideal; high productivity potential |
| **30–60** | 🟡 Moderate | Several factors need improvement; implement recommendations |
| **60–100** | 🔴 Critical | Major deviations; urgent lifestyle intervention recommended |

## Limitations & Future Work
- **Current:** Binary classification approach; future versions may add regression for continuous burnout scores
- **Generalization:** Model trained on 100K sample from 5M records; validation on new population recommended
- **Causal Claims:** Profile matching identifies correlations, not causal relationships; use results for awareness, not diagnosis
- **Dataset Dependency:** Results sensitive to input data quality; missing values handled via list-wise deletion

## Acknowledgements
- Dataset: Large-scale digital burnout & productivity survey (5M records, 34 features)
- Methodology: Profile matching (decision support systems) + Machine Learning ensemble approaches
- Framework: Scikit-learn (ML pipeline), pandas (data engineering), ipywidgets (interactive UI)

## License
This project is for educational, research, and personal wellness analysis purposes only. Use results as guidance only, not medical/professional advice. Consult professionals for serious burnout concerns.

## Contact & Support
For questions, improvements, or bug reports, please open an issue on GitHub or contact the project maintainer.

---

**Last Updated:** May 2026  
**Version:** 1.0  
**Status:** Active Development
