import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# =========================
# 🎨 PAGE
# =========================
st.set_page_config(page_title="Placement Predictor", layout="centered")

st.title("🎓 Smart Placement Predictor")
st.caption("Prediction • Company Fit • Roadmap • Improvement Guide")

# =========================
# 📂 LOAD DATA
# =========================
df = pd.read_csv(r"C:\clg\placementdata.csv")
df.columns = df.columns.str.strip()

if 'Workshops/Certifications' in df.columns:
    df.rename(columns={'Workshops/Certifications': 'Workshops'}, inplace=True)

target = 'PlacementStatus'
df[target] = df[target].map({'Placed':1,'NotPlaced':0,'Yes':1,'No':0})

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].map({'Yes':1,'No':0}).fillna(df[col])

X = df.drop([target,'StudentID'], axis=1, errors='ignore')
y = df[target]

# =========================
# 🤖 MODEL
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================
# 📊 METRICS
# =========================
if st.button("📊 Check Model Accuracy"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    st.write(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

# =========================
# 🎛 INPUT
# =========================
st.subheader("📊 Enter Your Details")

cgpa = st.slider("CGPA", 5.0, 10.0, 7.0)
internships = st.slider("Internships", 0, 5, 1)
projects = st.slider("Projects", 0, 5, 2)
workshops = st.slider("Workshops", 0, 5, 1)
aptitude = st.slider("Aptitude Score", 0, 100, 70)
softskills = st.slider("Soft Skills", 1, 10, 6)
ssc = st.slider("SSC Marks", 50, 100, 70)
hsc = st.slider("HSC Marks", 50, 100, 70)

placement_training = st.selectbox("Placement Training", ["No","Yes"])
extra = st.selectbox("Extracurricular", ["No","Yes"])

minor = st.selectbox("Domain", [
    "Data Science","Web Development","AI/ML",
    "Cyber Security","Software Testing","Cloud Computing"
])

placement_training = 1 if placement_training=="Yes" else 0
extra = 1 if extra=="Yes" else 0

# =========================
# 🏢 COMPANY
# =========================
company = st.selectbox("Target Company",
                       ["Google","Amazon","Microsoft","TCS","Infosys","Wipro"])

rubrics = {
    "Google": {"cgpa":8,"projects":4,"internships":2,"aptitude":80,"softskills":7},
    "Amazon": {"cgpa":7.5,"projects":3,"internships":1,"aptitude":75,"softskills":6},
    "Microsoft": {"cgpa":7.5,"projects":3,"internships":1,"aptitude":75,"softskills":7},
    "TCS": {"cgpa":6.5,"projects":2,"internships":0,"aptitude":65,"softskills":6},
    "Infosys": {"cgpa":6.5,"projects":2,"internships":0,"aptitude":65,"softskills":6},
    "Wipro": {"cgpa":6,"projects":1,"internships":0,"aptitude":60,"softskills":6}
}

rubric = rubrics[company]

# =========================
# 🎯 PREDICT
# =========================
if st.button("🎯 Predict & Analyze"):

    user_data = {
        'CGPA': cgpa,
        'Internships': internships,
        'Projects': projects,
        'Workshops': workshops,
        'AptitudeTestScore': aptitude,
        'SoftSkillsRating': softskills,
        'ExtracurricularActivities': extra,
        'PlacementTraining': placement_training,
        'SSC_Marks': ssc,
        'HSC_Marks': hsc
    }

    user_df = pd.DataFrame([user_data])[X.columns]
    user_scaled = scaler.transform(user_df)

    pred = model.predict(user_scaled)[0]
    prob = model.predict_proba(user_scaled)[0][1]

    st.subheader("📊 Placement Probability")
    st.progress(prob)
    st.write(f"{prob*100:.2f}%")

    # =========================
    # 📈 LINE GRAPH
    # =========================
    st.subheader("📈 You vs Company")

    labels = ["CGPA","Projects","Internships","Aptitude","Soft Skills"]
    user_scores = [cgpa, projects*2, internships*3, aptitude/10, softskills]
    comp_scores = [rubric["cgpa"], rubric["projects"]*2,
                   rubric["internships"]*3, rubric["aptitude"]/10,
                   rubric["softskills"]]

    df_chart = pd.DataFrame({
        "You": user_scores,
        "Company": comp_scores
    }, index=labels)

    st.line_chart(df_chart)

    # =========================
    # 📊 SIDEBAR
    # =========================
    good, missing = [], []

    if cgpa >= rubric["cgpa"]: good.append("CGPA")
    else: missing.append("CGPA")

    if projects >= rubric["projects"]: good.append("Projects")
    else: missing.append("Projects")

    if internships >= rubric["internships"]: good.append("Internships")
    else: missing.append("Internships")

    if aptitude >= rubric["aptitude"]: good.append("Aptitude")
    else: missing.append("Aptitude")

    if softskills >= rubric["softskills"]: good.append("Soft Skills")
    else: missing.append("Soft Skills")

    st.sidebar.title("📊 Company Fit")
    st.sidebar.write("✅ Strengths:", good)
    st.sidebar.write("⚠️ Missing:", missing)

    fit = int((len(good)/(len(good)+len(missing)))*100)
    st.sidebar.progress(fit/100)
    st.sidebar.write(f"{fit}% match with {company}")

    # =========================
    # ⚠️ IMPROVEMENTS + LINKS + TIME
    # =========================
    st.subheader("⚠️ Improvements")

    improvements, links, time_needed = [], [], []

    if cgpa < rubric["cgpa"]:
        improvements.append("Improve CGPA")
        links.append("https://www.youtube.com/@GateSmashers")
        time_needed.append("2-3 months")

    if projects < rubric["projects"]:
        improvements.append("Build Projects")
        links.append("https://www.youtube.com/@freecodecamp")
        time_needed.append("1-2 months")

    if internships < rubric["internships"]:
        improvements.append("Gain Internship")
        links.append("https://internshala.com")
        time_needed.append("1-3 months")

    if aptitude < rubric["aptitude"]:
        improvements.append("Improve Aptitude")
        links.append("https://www.youtube.com/@CareerRide")
        time_needed.append("3-4 weeks")

    if softskills < rubric["softskills"]:
        improvements.append("Improve Communication")
        links.append("https://www.youtube.com/@Charismaoncommand")
        time_needed.append("2-4 weeks")

    for i in range(len(improvements)):
        st.error(improvements[i])
        st.write("🔗", links[i])
        st.write("⏳", time_needed[i])

    # =========================
    # 📅 ROADMAP
    # =========================
    st.subheader("📅 4-Week Roadmap")

    roadmap = [
        ("Week 1", ["Revise basics", "Start aptitude"]),
        ("Week 2", ["Build project", "Practice coding"]),
        ("Week 3", ["Apply internships", "Mock interviews"]),
        ("Week 4", ["Revise + Apply jobs"])
    ]

    for week, tasks in roadmap:
        st.markdown(f"### {week}")
        for t in tasks:
            st.write("•", t)

    # =========================
    # 🔗 LINKEDIN
    # =========================
    st.subheader("🔗 Networking")

    link = f"https://www.linkedin.com/search/results/people/?keywords={company}%20engineer"
    st.markdown(f"[👉 Connect with {company} employees]({link})")

    # =========================
    # 💾 DOWNLOAD
    # =========================
    report = pd.DataFrame({
        "Improvement": improvements,
        "Link": links,
        "Time": time_needed
    })

    st.download_button(
        "📥 Download Report",
        report.to_csv(index=False),
        "placement_report.csv"
    )
