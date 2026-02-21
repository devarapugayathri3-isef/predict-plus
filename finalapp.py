import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="PREDICT+", layout="wide")

# Custom Styling
st.markdown("""
<style>
.big-title {
    font-size:48px !important;
    font-weight:700;
    color:#1f4e79;
}
.section-title {
    font-size:28px !important;
    font-weight:600;
    color:#2e7d32;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# EXECUTIVE SUMMARY
# --------------------------------------------------

st.markdown('<p class="big-title">PREDICT+</p>', unsafe_allow_html=True)

st.markdown("""
## Executive Summary

PREDICT+ is a biomedical engineering prototype that demonstrates how 
maternal wellness behaviors can be computationally modeled to produce 
an interpretable Fetal Comfort Score (FCS).

Using synthetic data and regression modeling, this dashboard illustrates 
how sleep, hydration, stress, activity, fetal movement, gestational age, 
and BMI collectively influence fetal comfort in a simulated environment.

The system emphasizes accessibility, interpretability, and ethical design.
""")

st.divider()

st.markdown("""
## Why This Matters

Maternal wellness significantly influences fetal development outcomes.  
Most monitoring systems are reactive rather than predictive.

This prototype demonstrates how wearable-compatible wellness metrics 
could be integrated into a computational framework to:

• Improve early risk awareness  
• Enhance patient education  
• Support preventative maternal care  
• Increase accessibility to data-driven insights  

This work represents a conceptual step toward ethical, interpretable 
maternal–fetal health modeling.
""")

st.divider()

# --------------------------------------------------
# DATA GENERATION
# --------------------------------------------------

@st.cache_data
def generate_dataset(n=1000):
    np.random.seed(42)

    data = pd.DataFrame({
        "sleep": np.random.uniform(4, 10, n),
        "hydration": np.random.uniform(1, 4, n),
        "stress": np.random.uniform(0, 10, n),
        "activity": np.random.uniform(0, 90, n),
        "movement": np.random.uniform(5, 40, n),
        "gestation": np.random.uniform(5, 40, n),
        "bmi": np.random.uniform(18, 35, n)
    })

    data["FCS_true"] = (
        0.22*(data["sleep"]/10) +
        0.15*(data["hydration"]/4) +
        0.20*(1-data["stress"]/10) +
        0.13*(data["activity"]/90) +
        0.15*(data["movement"]/40) +
        0.05*(data["gestation"]/40) +
        0.10*(1-abs(data["bmi"]-22)/15)
    )

    data["Risk_Category"] = pd.cut(
        data["FCS_true"],
        bins=[0, 0.5, 0.75, 1],
        labels=["High Risk", "Moderate Risk", "Low Risk"]
    )

    return data

data = generate_dataset()

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

X = data.drop(["FCS_true", "Risk_Category"], axis=1)
y = data["FCS_true"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)
MAE = mean_absolute_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

# Classification
X_class = X
y_class = data["Risk_Category"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(Xc_train, yc_train)
classification_accuracy = classifier.score(Xc_test, yc_test)

# --------------------------------------------------
# INTERACTIVE PANEL
# --------------------------------------------------

st.markdown("## Interactive Simulation Panel")

sleep = st.slider("Sleep (hours)", 0.0, 12.0, 7.0)
hydration = st.slider("Hydration (liters/day)", 0.0, 5.0, 2.5)
stress = st.slider("Stress Level (0-10)", 0, 10, 5)
activity = st.slider("Physical Activity (minutes)", 0, 120, 30)
movement = st.slider("Fetal Movement (kicks/day)", 0, 50, 20)
gestation = st.slider("Gestational Age (weeks)", 1, 40, 20)
bmi = st.slider("BMI", 15.0, 40.0, 25.0)

input_df = pd.DataFrame([{
    "sleep": sleep,
    "hydration": hydration,
    "stress": stress,
    "activity": activity,
    "movement": movement,
    "gestation": gestation,
    "bmi": bmi
}])

FCS = reg_model.predict(input_df)[0]

st.divider()

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

st.subheader("🩺 Fetal Comfort Score")

if FCS >= 0.75:
    st.success(f"Optimal Comfort — Score: {FCS:.2f}")
elif FCS >= 0.50:
    st.warning(f"Moderate Comfort — Score: {FCS:.2f}")
else:
    st.error(f"Suboptimal Comfort — Score: {FCS:.2f}")

st.subheader("⚠️ Predicted Risk Category")
risk_prediction = classifier.predict(input_df)[0]
st.write(risk_prediction)

st.divider()

# --------------------------------------------------
# RESEARCH PANEL
# --------------------------------------------------

st.markdown("## Advanced Research & Model Validation")

show_research = st.toggle("Show Research Panel")

if show_research:

    st.subheader("📊 Correlation Matrix")
    corr_matrix = data.drop("Risk_Category", axis=1).corr()
    st.dataframe(corr_matrix)

    fig, ax = plt.subplots()
    cax = ax.matshow(corr_matrix, cmap="coolwarm")
    fig.colorbar(cax)
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    st.pyplot(fig)

    st.subheader("📈 Model Validation Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error (MAE)", f"{MAE:.4f}")
    col2.metric("R² Score", f"{R2:.4f}")
    st.metric("Classification Accuracy", f"{classification_accuracy:.3f}")

    cv_scores = cross_val_score(classifier, X_class, y_class, cv=5)
    st.write("5-Fold Cross Validation Accuracy:", round(cv_scores.mean(), 3))

    st.subheader("Model-Learned Feature Importance")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Learned Weight": reg_model.coef_
    })
    st.bar_chart(coef_df.set_index("Feature"))

    st.subheader("📅 30-Day Simulated Trend")
    days = 30
    trend_data = generate_dataset(days)
    trend_X = trend_data.drop(["FCS_true", "Risk_Category"], axis=1)
    trend_FCS = reg_model.predict(trend_X)
    trend_df = pd.DataFrame({
        "Day": range(1, days+1),
        "FCS": trend_FCS
    })
    st.line_chart(trend_df.set_index("Day"))

    st.subheader("Engineering Design Framework")
    st.markdown("""
1. Identify maternal wellness variables  
2. Construct synthetic physiological model  
3. Implement regression-based prediction  
4. Validate with train/test split and cross-validation  
5. Interpret feature contributions  
6. Assess ethical and clinical limitations  
""")

    st.subheader("Limitations")
    st.markdown("""
• All data is simulated  
• Not a medical diagnostic system  
• Real maternal-fetal dynamics are more complex  
• Intended for educational and engineering demonstration  
""")
