import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

@st.cache_resource
def load_model():
    np.random.seed(42)
    n = 1000
    data = {
        'age': np.random.randint(20,80,n),
        'gender': np.random.choice([0,1],n),
        'blood_pressure': np.random.randint(60,180,n),
        'glucose_level': np.random.randint(70,300,n),
        'cholesterol': np.random.randint(150,350,n),
        'bmi': np.round(np.random.uniform(15,45,n),1),
        'heart_rate': np.random.randint(50,110,n),
        'smoking': np.random.choice([0,1],n),
        'family_history': np.random.choice([0,1],n),
    }
    def disease(r):
        s = sum([r['glucose_level']>140, r['glucose_level']>140,
                 r['blood_pressure']>130, r['bmi']>30,
                 r['family_history']==1, r['age']>50,
                 r['cholesterol']>240, r['smoking']==1])
        if s>=5: return 'Diabetes'
        elif r['blood_pressure']>140 and r['cholesterol']>240 and r['age']>45: return 'Heart Disease'
        elif r['bmi']>35 and r['blood_pressure']>130: return 'Hypertension'
        elif s>=3: return 'At Risk'
        else: return 'Healthy'
    df = pd.DataFrame(data)
    df['disease'] = df.apply(disease, axis=1)
    X = df.drop('disease', axis=1)
    le = LabelEncoder()
    y = le.fit_transform(df['disease'])
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(Xs, y)
    return m, sc, le

model, scaler, le = load_model()

INFO = {
    "Diabetes":      {"icon":"🩸","color":"#e74c3c","risk":"High",   "advice":"Monitor blood sugar. Low-sugar diet. See endocrinologist."},
    "Heart Disease": {"icon":"❤️","color":"#c0392b","risk":"High",   "advice":"Reduce cholesterol, quit smoking, see cardiologist."},
    "Hypertension":  {"icon":"🔴","color":"#e67e22","risk":"Medium", "advice":"Reduce salt, manage stress, monitor BP daily."},
    "At Risk":       {"icon":"⚠️","color":"#f39c12","risk":"Medium", "advice":"Healthier habits needed. Regular checkups recommended."},
    "Healthy":       {"icon":"✅","color":"#27ae60","risk":"Low",    "advice":"Great! Keep up your healthy lifestyle."}
}

st.set_page_config(page_title="AI Healthcare", page_icon="🏥")
st.markdown("<h1 style='text-align:center'>🏥 AI Smart Healthcare</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray'>Disease Prediction System</p>", unsafe_allow_html=True)
st.markdown("---")

c1, c2 = st.columns(2)
with c1:
    age            = st.number_input("Age (years)", 1, 120, 35)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", 40, 250, 120)
    cholesterol    = st.number_input("Cholesterol (mg/dL)", 100, 500, 190)
    heart_rate     = st.number_input("Heart Rate (bpm)", 30, 200, 75)
    smoking        = st.selectbox("Smoking", ["No","Yes"])
with c2:
    gender         = st.selectbox("Gender", ["Female","Male"])
    glucose_level  = st.number_input("Glucose Level (mg/dL)", 50, 500, 95)
    bmi            = st.number_input("BMI", 10.0, 60.0, 22.5)
    family_history = st.selectbox("Family History", ["No","Yes"])

st.markdown("---")
if st.button("🔍 Predict Disease", use_container_width=True, type="primary"):
    feat = [[age, 1 if gender=="Male" else 0, blood_pressure, glucose_level,
             cholesterol, bmi, heart_rate,
             1 if smoking=="Yes" else 0,
             1 if family_history=="Yes" else 0]]
    pred = le.inverse_transform(model.predict(scaler.transform(feat)))[0]
    conf = round(max(model.predict_proba(scaler.transform(feat))[0])*100, 1)
    info = INFO[pred]
    st.markdown(f"<h2 style='text-align:center;color:{info['color']}'>{info['icon']} {pred}</h2>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Confidence", f"{conf}%")
    m2.metric("Risk Level", info['risk'])
    st.info(f"💡 {info['advice']}")
    st.warning("⚠️ AI prediction only — not a medical diagnosis.")
