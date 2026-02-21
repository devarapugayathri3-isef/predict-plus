import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="PREDICT+", layout="centered")

# --------------------------------------------------
# Generate Synthetic Dataset (Only Once)
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
# Regression Model
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

# --------------------------------------------------
# Classification Model
# --------------------------------------------------
X_class = X
y_class = data["Risk_Category"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(Xc_train, yc_train)
classification_accuracy = classifier.score(Xc_test, yc_test)

# --------------------------------------------------
# USER INTERFACE
# --------------------------------------------------

st.title("PREDICT+")
st.subheader("Maternal–Fetal Wellness Computational Prototype")

st.markdown("""
### About This Tool

PREDICT+ is an educational computational dashboard that models a **Fetal Comfort Score (FCS)** 
based on seven measurable maternal wellness factors:

• Sleep duration  
• Hydration levels  
• Stress perception  
• Physical activity  
• Fetal movement  
• Gestational age  
• Maternal BMI  

The system uses simulated biomedical data to demonstrate how daily lifestyle patterns 
may influence overall fetal comfort in a conceptual modeling framework.

This tool is **not a medical device** and does not provide medical advice. 
It is designed for educational and engineering demonstration purposes only.
""")

st.markdown("""
### How to Use

Adjust the sliders below to simulate different maternal wellness scenarios.
The Fetal Comfort Score will update automatically in real time.
""")

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
    st.success(f"🟢 Optimal Comfort — Score: {FCS:.2f}")
elif FCS >= 0.50:
    st.warning(f"🟡 Moderate Comfort — Score: {FCS:.2f}")
else:
    st.error(f"🔴 Suboptimal Comfort — Score: {FCS:.2f}")

risk_prediction = classifier.predict(input_df)[0]

st.subheader("Predicted Risk Category")
st.write(risk_prediction)

st.markdown("""
**Risk Category Interpretation**

The risk classification model groups simulated observations into:

• High Risk  
• Moderate Risk  
• Low Risk  

This classification is generated using logistic regression trained on synthetic data.
It demonstrates how multiple maternal variables can collectively influence categorical risk levels.
""")

st.markdown("""
**What does this score mean?**

The Fetal Comfort Score (FCS) ranges from 0 to 1.

• **0.75 – 1.00** → Optimal simulated comfort  
• **0.50 – 0.74** → Moderate comfort  
• **Below 0.50** → Suboptimal simulated conditions  

The score is calculated using a weighted combination of normalized wellness inputs.
Higher sleep, hydration, and fetal movement — and lower stress — generally increase the score.
""")

# --------------------------------------------------
# RESEARCH PANEL (Hidden)
# --------------------------------------------------

show_research = st.toggle("Show Research & Model Validation Panel")

if show_research:

    st.subheader("Model Validation Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error (MAE)", f"{MAE:.4f}")
    col2.metric("R² Score", f"{R2:.4f}")

    st.metric("Classification Accuracy", f"{classification_accuracy:.3f}")

    cv_scores = cross_val_score(classifier, X_class, y_class, cv=5)
    st.write("5-Fold Cross Validation Accuracy:", round(cv_scores.mean(), 3))

    # ---- Feature Importance ----
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Learned Weight": reg_model.coef_
    })

    st.subheader("Model-Learned Feature Importance")
    st.bar_chart(coef_df.set_index("Feature"))

    # ---- Interpretation ----
    st.markdown("""
### Model Interpretation

The regression model shows how strongly each wellness factor contributes 
to predicted fetal comfort.

The high R² value reflects that the model was trained on synthetic data 
generated from a known physiological relationship. This demonstrates model stability 
rather than clinical prediction accuracy.

Cross-validation confirms that the classification model performs consistently 
across multiple data splits.

### Limitations

• All data used in this system is simulated  
• The model does not account for medical complications  
• This tool is educational and not diagnostic  
• Real-world maternal-fetal dynamics are more complex  

Future work could include integration with wearable health tracking devices 
or real longitudinal datasets under clinical supervision.
""")

    st.subheader("Model-Learned Feature Importance")
    st.bar_chart(coef_df.set_index("Feature"))
