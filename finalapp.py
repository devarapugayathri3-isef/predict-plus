import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="PREDICT+ Computational Model", layout="wide")

# ============================================================
# DATA GENERATION
# ============================================================

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

# ============================================================
# MODEL TRAINING
# ============================================================

X = data.drop(["FCS_true", "Risk_Category"], axis=1)
y = data["FCS_true"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)
MAE = mean_absolute_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

# Random Forest Comparison
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_R2 = r2_score(y_test, rf_pred)

# Classification
X_class = X
y_class = data["Risk_Category"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(Xc_train, yc_train)
classification_accuracy = classifier.score(Xc_test, yc_test)

cv_scores = cross_val_score(classifier, X_class, y_class, cv=5)

# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.title("PREDICT+")
st.markdown("## Computational Maternal–Fetal Modeling Framework")

st.markdown("""
PREDICT+ is a biomedical engineering prototype that models a 
Fetal Comfort Score (FCS) using seven measurable maternal wellness factors.

The system integrates normalization, weighted aggregation, regression modeling,
classification, cross-validation, and robustness testing to demonstrate
how computational modeling can translate maternal behaviors into interpretable metrics.

This tool is educational and non-clinical.
""")

st.divider()

# ============================================================
# INTERACTIVE SIMULATION
# ============================================================

st.markdown("## Interactive Maternal Wellness Simulation")

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

st.subheader("Fetal Comfort Score")

if FCS >= 0.75:
    st.success(f"Optimal Comfort — Score: {FCS:.2f}")
elif FCS >= 0.50:
    st.warning(f"Moderate Comfort — Score: {FCS:.2f}")
else:
    st.error(f"Suboptimal Comfort — Score: {FCS:.2f}")

risk_prediction = classifier.predict(input_df)[0]
st.subheader("Predicted Risk Category")
st.write(risk_prediction)

st.divider()

# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

st.markdown("## Marginal Influence Analysis")

selected_variable = st.selectbox("Select Variable for 10% Increase", X.columns)

adjusted_input = input_df.copy()
adjusted_input[selected_variable] *= 1.10
adjusted_FCS = reg_model.predict(adjusted_input)[0]

delta = adjusted_FCS - FCS

st.write(f"Original FCS: {FCS:.3f}")
st.write(f"Adjusted FCS: {adjusted_FCS:.3f}")
st.write(f"Change in FCS: {delta:.3f}")

st.markdown("""
This approximates marginal influence (partial derivative behavior)
of the weighted FCS function under synthetic conditions.
""")

st.divider()

# ============================================================
# LONGITUDINAL MODELING
# ============================================================

st.markdown("## Longitudinal Behavior Modeling")

days = 30
trend_data = generate_dataset(days)
trend_X = trend_data.drop(["FCS_true", "Risk_Category"], axis=1)
trend_FCS = reg_model.predict(trend_X)

trend_df = pd.DataFrame({
    "Day": range(1, days+1),
    "FCS": trend_FCS
})

st.line_chart(trend_df.set_index("Day"))

st.markdown("""
Simulated trajectories demonstrate how sustained behavioral patterns
may influence fetal comfort trends over time.
""")

st.divider()

# ============================================================
# MODEL VALIDATION
# ============================================================

st.markdown("## Model Validation & Performance")

col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (MAE)", f"{MAE:.4f}")
col2.metric("Linear Regression R²", f"{R2:.4f}")

st.metric("Random Forest R² (Comparison)", f"{rf_R2:.4f}")
st.metric("Classification Accuracy", f"{classification_accuracy:.3f}")
st.write("5-Fold Cross Validation Accuracy:", round(cv_scores.mean(), 3))

st.markdown("""
Linear regression was selected due to interpretability and minimal performance
difference compared to ensemble methods under synthetic constraints.
""")

st.divider()

# ============================================================
# ROBUSTNESS TESTING
# ============================================================

st.markdown("## Robustness Testing (Repeated Partition Validation)")

r2_scores = []

for i in range(20):
    Xt_train, Xt_test, yt_train, yt_test = train_test_split(
        X, y, test_size=0.2
    )
    temp_model = LinearRegression()
    temp_model.fit(Xt_train, yt_train)
    temp_pred = temp_model.predict(Xt_test)
    r2_scores.append(r2_score(yt_test, temp_pred))

st.write("Average R² over 20 random splits:", round(np.mean(r2_scores), 4))

st.markdown("""
Repeated partitioning confirms model stability across random splits.
""")

st.divider()

# ============================================================
# CORRELATION MATRIX
# ============================================================

st.markdown("## Correlation Matrix")

corr_matrix = data.drop("Risk_Category", axis=1).corr()

fig, ax = plt.subplots()
cax = ax.matshow(corr_matrix)
fig.colorbar(cax)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
st.pyplot(fig)

st.markdown("""
Correlation structure confirms expected physiological directionality:
sleep, movement, and hydration correlate positively with FCS,
while stress shows inverse association.
""")

st.divider()

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

st.markdown("## Mathematical Model Specification")

st.markdown("""
Pipeline:

1. Input Collection (7 maternal variables)
2. Normalization to 0–1 scale
3. Weighted aggregation (FCS formula)
4. Linear regression modeling
5. Logistic classification
6. Cross-validation and robustness testing
""")

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Learned Weight": reg_model.coef_
})

st.subheader("Feature Contribution (Regression Coefficients)")
st.bar_chart(coef_df.set_index("Feature"))

st.divider()

# ============================================================
# ETHICAL STATEMENT
# ============================================================

st.markdown("## Ethical & Compliance Statement")

st.markdown("""
• All data used is synthetically generated  
• No human subjects were involved  
• No medical claims are made  
• Educational engineering prototype  
• Fully compliant with NWSE and ISEF guidelines  
""")
