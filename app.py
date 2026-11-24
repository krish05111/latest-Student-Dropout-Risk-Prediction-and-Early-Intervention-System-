import os

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# -----------------------------------
# PATHS AND MODEL INPUTS
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "dropout_model.pkl")
COMPARISON_PATH = os.path.join(BASE_DIR, "models", "model_comparison.csv")
REVIEW_PATH = os.path.join(BASE_DIR, "data", "advisor_reviews.csv")

FEATURES = [
    "attendance",
    "marks",
    "study_hours",
    "failed_subjects",
    "assignments_completed",
]

FEATURE_LABELS = {
    "attendance": "Attendance",
    "marks": "Average Marks",
    "study_hours": "Daily Study Hours",
    "failed_subjects": "Failed Subjects",
    "assignments_completed": "Assignments Completed",
}

RISK_COLORS = {
    "LOW": "#60A5FA",
    "MEDIUM": "#1D4ED8",
    "HIGH": "#0B1F44",
}

RISK_HINTS = {
    "LOW": "Probability below 30%",
    "MEDIUM": "Probability from 30% to below 60%",
    "HIGH": "Probability of 60% or higher",
}

PAGES = [
    "Dashboard",
    "Student Risk Prediction",
    "Explainable AI",
    "Student Records",
    "Advisor Panel",
    "Model Performance",
]

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Student Dropout Risk Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "prediction" not in st.session_state:
    st.session_state.prediction = None

# -----------------------------------
# CUSTOM STYLE
# -----------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: #F4F8FC;
        color: #0B1F44;
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1240px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1F44 0%, #12315C 70%, #1E4B9C 140%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #EAF2FF;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border: 1px solid #E2EAF5;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(11, 31, 68, 0.06);
    }

    .page-header h1 {
        margin: 0;
        color: #0B1F44;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }

    .page-header p {
        margin: 0.4rem 0 0 0;
        color: #4B6180;
        font-size: 1rem;
        line-height: 1.5;
    }

    .metric-card, .risk-card {
        background: #FFFFFF;
        border: 1px solid #E2EAF5;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(11, 31, 68, 0.06);
        padding: 1.05rem 1.15rem;
        min-height: 124px;
        box-sizing: border-box;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .metric-card:hover, .risk-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 28px rgba(11, 31, 68, 0.12);
    }

    .metric-card h3, .risk-card span {
        margin: 0;
        color: #5C7190;
        font-size: 0.82rem;
        font-weight: 650;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .metric-card p {
        margin: 0.35rem 0 0.2rem 0;
        color: #0B1F44;
        font-size: 1.9rem;
        font-weight: 720;
        line-height: 1.1;
    }

    .metric-card small, .risk-card p {
        color: #5C7190;
        font-size: 0.9rem;
    }

    .metric-card.accent {
        background: linear-gradient(180deg, #FFFFFF 0%, #E8F2FC 100%);
    }

    .risk-card.active {
        background: #E8F2FC;
        border: 2px solid #1D4ED8;
    }

    .risk-card strong {
        display: block;
        margin-top: 0.45rem;
        color: #0B1F44;
        font-size: 1.15rem;
    }

    .brand-kicker {
        color: #93C5FD;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 650;
        border: 1px solid #D5E3F5;
        transition: background 0.18s ease, transform 0.18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }

    .stButton > button[kind="primary"] {
        background: #0B1F44;
        color: #FFFFFF;
        border: none;
    }

    .stButton > button[kind="primary"]:hover {
        background: #1D4ED8;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------
# LOAD EXISTING DATA AND MODEL
# -----------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_student_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_model_comparison():
    return pd.read_csv(COMPARISON_PATH)


def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "The trained model was not found at models/dropout_model.pkl. "
            "Run train_model.py, then start the app again."
        )
        st.stop()
    return load_model()


def get_student_data():
    if not os.path.exists(DATA_PATH):
        st.error(
            "The student dataset was not found at data/student_data.csv."
        )
        st.stop()
    return load_student_data().copy()


def get_model_comparison():
    if not os.path.exists(COMPARISON_PATH):
        return None
    return load_model_comparison().copy()


# -----------------------------------
# PREDICTION LOGIC
# The thresholds and checks below match the original application.
# -----------------------------------

def classify_risk(probability_percent):
    if probability_percent < 30:
        return "LOW"
    elif probability_percent < 60:
        return "MEDIUM"
    else:
        return "HIGH"


def identify_risk_factors(
    attendance,
    marks,
    study_hours,
    failed_subjects,
    assignments_completed,
):
    risk_factors = []

    if attendance < 60:
        risk_factors.append("Low attendance")

    if marks < 50:
        risk_factors.append("Low academic marks")

    if study_hours < 2:
        risk_factors.append("Low study hours")

    if failed_subjects >= 2:
        risk_factors.append("Multiple failed subjects")

    if assignments_completed < 5:
        risk_factors.append("Low assignment completion")

    return risk_factors


def recommend_interventions(
    attendance,
    marks,
    study_hours,
    failed_subjects,
    assignments_completed,
    risk_level,
):
    recommendations = []

    if attendance < 60:
        recommendations.append(
            "Schedule attendance counselling with the student."
        )

    if marks < 50:
        recommendations.append(
            "Provide additional academic tutoring."
        )

    if study_hours < 2:
        recommendations.append(
            "Help the student create a regular study schedule."
        )

    if failed_subjects >= 2:
        recommendations.append(
            "Arrange academic mentoring and subject support."
        )

    if assignments_completed < 5:
        recommendations.append(
            "Follow up with the student regarding pending assignments."
        )

    if risk_level == "HIGH":
        recommendations.append(
            "Schedule an immediate meeting with an academic advisor."
        )

    if len(recommendations) == 0:
        recommendations.append("Continue regular monitoring.")

    return recommendations


def predict_student(
    model,
    attendance,
    marks,
    study_hours,
    failed_subjects,
    assignments_completed,
    source_label="Manual entry",
):
    """Run the saved model and keep the original risk rules."""
    student_data = pd.DataFrame(
        [[
            attendance,
            marks,
            study_hours,
            failed_subjects,
            assignments_completed,
        ]],
        columns=FEATURES,
    )

    probability = model.predict_proba(student_data)[0][1]
    probability_percent = float(probability * 100)
    risk_level = classify_risk(probability_percent)
    risk_factors = identify_risk_factors(
        attendance,
        marks,
        study_hours,
        failed_subjects,
        assignments_completed,
    )
    recommendations = recommend_interventions(
        attendance,
        marks,
        study_hours,
        failed_subjects,
        assignments_completed,
        risk_level,
    )

    return {
        "attendance": int(attendance),
        "marks": int(marks),
        "study_hours": int(study_hours),
        "failed_subjects": int(failed_subjects),
        "assignments_completed": int(assignments_completed),
        "probability": float(probability),
        "probability_percent": probability_percent,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommendations": recommendations,
        "source_label": source_label,
    }


def score_students(data, model):
    """Score every row in the existing student CSV with the saved model."""
    scored = data.copy()
    probabilities = model.predict_proba(scored[FEATURES])[:, 1]
    probability_percent = probabilities * 100

    scored.insert(
        0,
        "Student",
        [f"Student {number}" for number in range(1, len(scored) + 1)],
    )
    scored["risk_probability"] = probability_percent
    scored["risk_level"] = [
        classify_risk(value) for value in probability_percent
    ]
    scored["recorded_outcome"] = scored["dropout"].map(
        {0: "No Dropout", 1: "Dropout"}
    )
    return scored


def saved_model_name(model):
    names = {
        "DummyClassifier": "Baseline",
        "LogisticRegression": "Logistic Regression",
        "RandomForestClassifier": "Random Forest",
    }
    return names.get(type(model).__name__, type(model).__name__)


def feature_importance_table(model):
    """Use the same importance measure as feature_importance.py when possible."""
    if hasattr(model, "coef_"):
        importance = abs(model.coef_[0])
        signed = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        signed = None
    else:
        return None

    table = pd.DataFrame({
        "Feature": [FEATURE_LABELS[name] for name in FEATURES],
        "Feature key": FEATURES,
        "Importance": importance,
    })

    if signed is not None:
        table["Coefficient"] = signed
        table["Direction"] = [
            "Higher values raise predicted risk"
            if value > 0
            else "Higher values lower predicted risk"
            for value in signed
        ]

    return table.sort_values("Importance", ascending=False)


def linear_contributions(model, prediction):
    """Break a logistic-regression score into intercept plus feature contributions."""
    if not hasattr(model, "coef_"):
        return None, None

    rows = []
    for feature, coefficient in zip(FEATURES, model.coef_[0]):
        value = float(prediction[feature])
        rows.append({
            "Feature": FEATURE_LABELS[feature],
            "Student value": value,
            "Coefficient": float(coefficient),
            "Contribution": float(coefficient * value),
        })

    contributions = pd.DataFrame(rows)
    intercept = float(model.intercept_[0])
    return contributions, intercept


def build_explanation(prediction):
    percent = prediction["probability_percent"]
    level = prediction["risk_level"]
    reason = RISK_HINTS[level].lower()

    explanation = (
        f"{prediction['source_label']} has a model dropout probability of "
        f"{percent:.2f}%. The result is {level} risk because the probability "
        f"is {reason}."
    )

    if prediction["risk_factors"]:
        factor_text = ", ".join(prediction["risk_factors"])
        explanation += f" Rule-based checks flagged: {factor_text}."
    else:
        explanation += " Rule-based checks did not flag a major risk factor."

    return explanation


def save_advisor_review(prediction, advisor_decision, advisor_notes):
    review_data = pd.DataFrame(
        [{
            "attendance": prediction["attendance"],
            "marks": prediction["marks"],
            "study_hours": prediction["study_hours"],
            "failed_subjects": prediction["failed_subjects"],
            "assignments_completed": prediction["assignments_completed"],
            "risk_level": prediction["risk_level"],
            "risk_probability": round(prediction["probability_percent"], 2),
            "advisor_decision": advisor_decision,
            "advisor_notes": advisor_notes,
        }]
    )

    if os.path.exists(REVIEW_PATH):
        review_data.to_csv(
            REVIEW_PATH,
            mode="a",
            header=False,
            index=False,
        )
    else:
        review_data.to_csv(REVIEW_PATH, index=False)


def load_reviews():
    if not os.path.exists(REVIEW_PATH):
        return None
    return pd.read_csv(REVIEW_PATH)


def compute_confusion_matrix(model):
    """Use the same split as evaluate_model.py and train_model.py."""
    data = get_student_data()
    features = data[FEATURES]
    target = data["dropout"]

    _x_train, x_test, _y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions)
    return matrix


def style_chart(fig, height=420):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Segoe UI, sans-serif", "color": "#0B1F44"},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        coloraxis_showscale=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E2EAF5")
    return fig


def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="page-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


def metric_card(label, value, note, accent=False):
    accent_class = " accent" if accent else ""
    st.markdown(
        f"""
        <div class="metric-card{accent_class}">
            <h3>{label}</h3>
            <p>{value}</p>
            <small>{note}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_indicator(level, active):
    state = "active" if active else ""
    badge = "Selected result" if active else "Risk band"
    st.markdown(
        f"""
        <div class="risk-card {state}">
            <span>{level} risk</span>
            <strong>{badge}</strong>
            <p>{RISK_HINTS[level]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_factor_list(prediction):
    if len(prediction["risk_factors"]) == 0:
        st.success("No major risk factors detected.")
    else:
        for factor in prediction["risk_factors"]:
            st.write(f"• {factor}")


def render_recommendations(prediction):
    for recommendation in prediction["recommendations"]:
        st.write(f"• {recommendation}")


# -----------------------------------
# SIDEBAR
# -----------------------------------

def render_sidebar():
    st.sidebar.markdown(
        """
        <div class="brand-kicker">Education analytics</div>
        <h2 style="margin-top:0; line-height:1.25;">Student Dropout Risk</h2>
        <p style="color:#D6E6FF; font-size:0.92rem;">
            Early intervention dashboard powered by the saved scikit-learn model.
        </p>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
    return page


def render_sidebar_status():
    prediction = st.session_state.prediction
    st.sidebar.write("")
    if prediction is None:
        st.sidebar.info("No prediction yet. Open Student Risk Prediction to score a student.")
    else:
        st.sidebar.success(
            f"{prediction['source_label']}: {prediction['risk_level']} risk "
            f"({prediction['probability_percent']:.2f}%)"
        )

    st.sidebar.caption(
        "Student Dropout Risk Prediction and Early Intervention System | "
        "AI Project Prototype"
    )


# -----------------------------------
# DASHBOARD
# -----------------------------------

def render_dashboard(model):
    page_header(
        "Dashboard",
        "Risk counts are produced by running the saved model on data/student_data.csv. "
        "No placeholder statistics are added.",
    )

    students = score_students(get_student_data(), model)
    total = len(students)
    high_count = int((students["risk_level"] == "HIGH").sum())
    medium_count = int((students["risk_level"] == "MEDIUM").sum())
    low_count = int((students["risk_level"] == "LOW").sum())

    card_columns = st.columns(4)
    cards = [
        ("Total students", total, "Rows in student_data.csv", False),
        ("High-risk students", high_count, "Probability of 60% or higher", True),
        ("Medium-risk students", medium_count, "Probability from 30% to below 60%", False),
        ("Low-risk students", low_count, "Probability below 30%", False),
    ]
    for column, (label, value, note, accent) in zip(card_columns, cards):
        with column:
            metric_card(label, value, note, accent)

    st.write("")
    chart_left, chart_right = st.columns([1, 1.35])

    with chart_left:
        with st.container(border=True):
            st.markdown("**Predicted risk distribution**")
            distribution = (
                students["risk_level"]
                .value_counts()
                .reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
                .reset_index()
            )
            distribution.columns = ["Risk level", "Students"]
            figure = go.Figure(
                go.Pie(
                    labels=distribution["Risk level"],
                    values=distribution["Students"],
                    hole=0.62,
                    marker={
                        "colors": [RISK_COLORS[level] for level in distribution["Risk level"]],
                        "line": {"color": "#FFFFFF", "width": 3},
                    },
                    textinfo="label+value",
                    hovertemplate="%{label}<br>%{value} students<br>%{percent}<extra></extra>",
                    sort=False,
                )
            )
            figure.update_layout(
                showlegend=True,
                annotations=[{
                    "text": f"<b>{total}</b><br>students",
                    "x": 0.5,
                    "y": 0.5,
                    "font": {"size": 18, "color": "#0B1F44"},
                    "showarrow": False,
                }],
            )
            style_chart(figure, height=430)
            st.plotly_chart(figure, width="stretch")

    with chart_right:
        with st.container(border=True):
            st.markdown("**Dropout probability by student**")
            probability_chart = px.bar(
                students,
                x="Student",
                y="risk_probability",
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                category_orders={"risk_level": ["LOW", "MEDIUM", "HIGH"]},
                hover_data=FEATURES,
                labels={
                    "risk_probability": "Dropout probability (%)",
                    "risk_level": "Risk level",
                },
            )
            probability_chart.add_hline(
                y=30,
                line_dash="dot",
                line_color="#60A5FA",
                annotation_text="30%",
            )
            probability_chart.add_hline(
                y=60,
                line_dash="dot",
                line_color="#0B1F44",
                annotation_text="60%",
            )
            probability_chart.update_yaxes(range=[0, 100])
            style_chart(probability_chart, height=430)
            st.plotly_chart(probability_chart, width="stretch")

    with st.expander("About the project"):
        st.write(
            """
            This is a basic AI-assisted prototype designed to identify
            students who may be at risk of dropping out.

            The system uses Machine Learning to analyze student-related
            factors and predict dropout risk.

            The system considers:

            • Attendance

            • Academic Marks

            • Daily Study Hours

            • Failed Subjects

            • Assignment Completion

            Based on the identified risk factors, the system provides
            early intervention recommendations.

            The final decision remains with the academic advisor,
            who can accept or override the AI recommendation.
            """
        )


# -----------------------------------
# STUDENT RISK PREDICTION
# -----------------------------------

def render_prediction(model):
    page_header(
        "Student Risk Prediction",
        "Enter the same five student factors used by the saved model. "
        "The result is stored so it remains available on the other pages.",
    )

    form_column, result_column = st.columns([1, 1.15], gap="large")

    with form_column:
        with st.container(border=True):
            st.markdown("**Student details**")
            st.caption("Adjust the inputs, then calculate risk with the saved model.")

            current_inputs = st.session_state.prediction or {}

            with st.form("prediction_form"):
                left, right = st.columns(2)

                with left:
                    attendance = st.slider(
                        "Attendance Percentage",
                        min_value=0,
                        max_value=100,
                        value=int(current_inputs.get("attendance", 75)),
                    )
                    marks = st.slider(
                        "Average Marks",
                        min_value=0,
                        max_value=100,
                        value=int(current_inputs.get("marks", 60)),
                    )
                    study_hours = st.slider(
                        "Daily Study Hours",
                        min_value=0,
                        max_value=12,
                        value=int(current_inputs.get("study_hours", 3)),
                    )

                with right:
                    failed_subjects = st.slider(
                        "Failed Subjects",
                        min_value=0,
                        max_value=10,
                        value=int(current_inputs.get("failed_subjects", 0)),
                    )
                    assignments_completed = st.slider(
                        "Assignments Completed",
                        min_value=0,
                        max_value=10,
                        value=int(current_inputs.get("assignments_completed", 5)),
                    )

                submitted = st.form_submit_button(
                    "Predict Risk",
                    width="stretch",
                    type="primary",
                )

            if submitted:
                st.session_state.prediction = predict_student(
                    model,
                    attendance,
                    marks,
                    study_hours,
                    failed_subjects,
                    assignments_completed,
                    source_label="Manual entry",
                )

    prediction = st.session_state.prediction

    with result_column:
        with st.container(border=True):
            st.markdown("**Prediction result**")
            if prediction is None:
                st.info(
                    "Run a prediction to see the risk level, probability, "
                    "and recommended intervention."
                )
            else:
                st.metric(
                    "Dropout Risk Probability",
                    f"{prediction['probability_percent']:.2f}%",
                )
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prediction["probability_percent"],
                        number={"suffix": "%", "font": {"color": "#0B1F44"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1},
                            "bar": {"color": RISK_COLORS[prediction["risk_level"]]},
                            "bgcolor": "#F4F8FC",
                            "steps": [
                                {"range": [0, 30], "color": "#DBEAFE"},
                                {"range": [30, 60], "color": "#93C5FD"},
                                {"range": [60, 100], "color": "#BFDBFE"},
                            ],
                            "threshold": {
                                "line": {"color": "#0B1F44", "width": 4},
                                "thickness": 0.8,
                                "value": prediction["probability_percent"],
                            },
                        },
                    )
                )
                style_chart(gauge, height=280)
                gauge.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 10})
                st.plotly_chart(gauge, width="stretch")

    if prediction is not None:
        st.write("")
        low_col, medium_col, high_col = st.columns(3)
        with low_col:
            risk_indicator("LOW", prediction["risk_level"] == "LOW")
        with medium_col:
            risk_indicator("MEDIUM", prediction["risk_level"] == "MEDIUM")
        with high_col:
            risk_indicator("HIGH", prediction["risk_level"] == "HIGH")

        st.write("")
        factor_column, action_column = st.columns(2)
        with factor_column:
            with st.container(border=True):
                st.markdown("**Identified risk factors**")
                render_factor_list(prediction)
        with action_column:
            with st.container(border=True):
                st.markdown("**Recommended early intervention**")
                render_recommendations(prediction)


# -----------------------------------
# EXPLAINABLE AI
# -----------------------------------

def render_explainable_ai(model):
    page_header(
        "Explainable AI",
        "See which inputs the saved model treats as most important, "
        "then read the factors behind an individual prediction.",
    )

    importance = feature_importance_table(model)
    if importance is None:
        st.warning("This saved model does not expose feature importance values.")
    else:
        with st.container(border=True):
            st.markdown("**Feature importance**")
            if hasattr(model, "coef_"):
                importance_note = (
                    "Importance is the absolute model coefficient, matching feature_importance.py. "
                    "A larger bar means the input has a stronger influence."
                )
            else:
                importance_note = (
                    "Importance comes from the saved model's feature importance scores. "
                    "A larger bar means the input has a stronger influence."
                )
            st.caption(importance_note)
            importance_chart = px.bar(
                importance,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#DBEAFE", "#1D4ED8", "#0B1F44"],
            )
            importance_chart.update_layout(
                yaxis={"categoryorder": "total ascending"},
                xaxis_title="Absolute coefficient",
                yaxis_title="",
            )
            style_chart(importance_chart, height=380)
            st.plotly_chart(importance_chart, width="stretch")

        if "Coefficient" in importance.columns:
            direction_table = importance[
                ["Feature", "Coefficient", "Direction"]
            ].copy()
            direction_table["Coefficient"] = direction_table["Coefficient"].round(4)
            st.dataframe(
                direction_table,
                hide_index=True,
                width="stretch",
            )

    st.write("")
    students = score_students(get_student_data(), model)
    choices = ["Latest prediction"] if st.session_state.prediction else []
    choices.extend(students["Student"].tolist())

    if not choices:
        st.info("Add student records or make a prediction to see an individual explanation.")
        return

    selected = st.selectbox("Explain a student", choices)
    if selected == "Latest prediction":
        prediction = st.session_state.prediction
    else:
        row = students[students["Student"] == selected].iloc[0]
        prediction = predict_student(
            model,
            row["attendance"],
            row["marks"],
            row["study_hours"],
            row["failed_subjects"],
            row["assignments_completed"],
            source_label=selected,
        )

    with st.container(border=True):
        st.markdown("**Why this prediction was made**")
        st.write(build_explanation(prediction))

    factor_column, contribution_column = st.columns(2)

    with factor_column:
        with st.container(border=True):
            st.markdown("**Individual risk factors**")
            input_table = pd.DataFrame({
                "Input": [FEATURE_LABELS[feature] for feature in FEATURES],
                "Value": [prediction[feature] for feature in FEATURES],
            })
            st.dataframe(input_table, hide_index=True, width="stretch")
            render_factor_list(prediction)

    with contribution_column:
        with st.container(border=True):
            st.markdown("**How the inputs push this prediction**")
            contributions, intercept = linear_contributions(model, prediction)
            if contributions is None:
                st.info(
                    "This model does not use linear coefficients. "
                    "Use the feature importance chart and the risk-factor list above."
                )
            else:
                log_odds = intercept + float(contributions["Contribution"].sum())
                st.caption(
                    "Each bar is the coefficient multiplied by this student's value. "
                    "Positive contributions push the score toward dropout. "
                    f"Starting point (intercept): {intercept:.2f}. "
                    f"Final log-odds: {log_odds:.2f}."
                )
                waterfall = go.Figure(
                    go.Waterfall(
                        x=["Starting point"]
                        + contributions["Feature"].tolist()
                        + ["Log-odds"],
                        y=[intercept]
                        + contributions["Contribution"].tolist()
                        + [log_odds],
                        measure=["absolute"]
                        + ["relative"] * len(contributions)
                        + ["total"],
                        decreasing={"marker": {"color": "#60A5FA"}},
                        increasing={"marker": {"color": "#0B1F44"}},
                        totals={"marker": {"color": "#1D4ED8"}},
                        connector={"line": {"color": "#CBD5E1"}},
                    )
                )
                style_chart(waterfall, height=420)
                waterfall.update_layout(showlegend=False)
                st.plotly_chart(waterfall, width="stretch")


# -----------------------------------
# STUDENT RECORDS
# -----------------------------------

def render_records(model):
    page_header(
        "Student Records",
        "Search the existing student dataset and filter it by the risk category "
        "assigned by the saved model.",
    )

    students = score_students(get_student_data(), model)
    search_column, filter_column = st.columns([2, 1])

    with search_column:
        query = st.text_input(
            "Search students",
            placeholder="Search by student number, marks, attendance, or outcome",
        )
    with filter_column:
        selected_levels = st.multiselect(
            "Risk category",
            ["LOW", "MEDIUM", "HIGH"],
            default=["LOW", "MEDIUM", "HIGH"],
        )

    if len(selected_levels) == 0:
        st.info("Choose at least one risk category to show matching students.")
        return

    filtered = students[students["risk_level"].isin(selected_levels)].copy()
    if query.strip():
        search_mask = filtered.astype(str).apply(
            lambda column: column.str.contains(query.strip(), case=False, na=False)
        ).any(axis=1)
        filtered = filtered[search_mask]

    st.caption(f"{len(filtered)} of {len(students)} students shown.")

    display = filtered.copy()
    display["risk_probability"] = display["risk_probability"].round(2)
    display = display.rename(columns={
        "attendance": "Attendance",
        "marks": "Average marks",
        "study_hours": "Study hours",
        "failed_subjects": "Failed subjects",
        "assignments_completed": "Assignments completed",
        "risk_level": "Risk level",
        "risk_probability": "Risk probability (%)",
        "recorded_outcome": "Recorded outcome",
    })
    visible_columns = [
        "Student",
        "Attendance",
        "Average marks",
        "Study hours",
        "Failed subjects",
        "Assignments completed",
        "Risk level",
        "Risk probability (%)",
        "Recorded outcome",
    ]

    st.dataframe(
        display[visible_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "Risk probability (%)": st.column_config.ProgressColumn(
                "Risk probability (%)",
                min_value=0,
                max_value=100,
                format="%.2f",
            )
        },
    )

    if filtered.empty:
        st.info("No students match this search.")
        return

    st.write("")
    with st.container(border=True):
        st.markdown("**Student details**")
        selected_student = st.selectbox(
            "Choose a record",
            filtered["Student"].tolist(),
        )
        row = filtered[filtered["Student"] == selected_student].iloc[0]
        detail = predict_student(
            model,
            row["attendance"],
            row["marks"],
            row["study_hours"],
            row["failed_subjects"],
            row["assignments_completed"],
            source_label=selected_student,
        )

        detail_columns = st.columns(4)
        with detail_columns[0]:
            metric_card("Risk level", detail["risk_level"], selected_student)
        with detail_columns[1]:
            metric_card(
                "Probability",
                f"{detail['probability_percent']:.2f}%",
                "Saved model output",
            )
        with detail_columns[2]:
            metric_card(
                "Recorded outcome",
                row["recorded_outcome"],
                "Label from student_data.csv",
            )
        with detail_columns[3]:
            metric_card(
                "Failed subjects",
                detail["failed_subjects"],
                f"Attendance {detail['attendance']}%",
            )

        factor_column, action_column = st.columns(2)
        with factor_column:
            st.markdown("**Risk factors**")
            render_factor_list(detail)
        with action_column:
            st.markdown("**Suggested intervention**")
            render_recommendations(detail)

        if st.button("Send this student to Advisor Panel", type="primary"):
            st.session_state.prediction = detail
            st.success(
                f"{selected_student} is ready for review. Open Advisor Panel to save a decision."
            )


# -----------------------------------
# ADVISOR PANEL
# -----------------------------------

def render_advisor():
    page_header(
        "Advisor Panel",
        "Review the current prediction, accept or override the recommendation, "
        "and save the note to data/advisor_reviews.csv.",
    )

    prediction = st.session_state.prediction

    if prediction is None:
        st.info(
            "Make a prediction, or send a student from Student Records, "
            "before saving an advisor review."
        )
    else:
        summary_columns = st.columns(3)
        with summary_columns[0]:
            metric_card("Student", prediction["source_label"], "Current review")
        with summary_columns[1]:
            metric_card("Risk level", prediction["risk_level"], "Model category")
        with summary_columns[2]:
            metric_card(
                "Probability",
                f"{prediction['probability_percent']:.2f}%",
                "Saved with the review",
            )

        st.write("")
        review_column, note_column = st.columns([1, 1.2])

        with review_column:
            with st.container(border=True):
                st.markdown("**Recommendation to review**")
                render_recommendations(prediction)
                st.markdown("**Risk factors**")
                render_factor_list(prediction)

        with note_column:
            with st.container(border=True):
                st.markdown("**Advisor decision**")
                advisor_decision = st.radio(
                    "Advisor Decision",
                    [
                        "Accept Recommendation",
                        "Override Recommendation",
                    ],
                )
                advisor_notes = st.text_area(
                    "Advisor Notes",
                    placeholder="Enter advisor comments here...",
                )

                if st.button("Save Advisor Review", type="primary", width="stretch"):
                    save_advisor_review(
                        prediction,
                        advisor_decision,
                        advisor_notes,
                    )
                    st.success("Advisor review saved successfully.")

    st.write("")
    st.markdown("**Saved reviews**")
    reviews = load_reviews()
    if reviews is None or reviews.empty:
        st.info("No advisor reviews have been saved yet.")
    else:
        st.dataframe(reviews, hide_index=True, width="stretch")


# -----------------------------------
# MODEL PERFORMANCE
# -----------------------------------

def render_performance(model):
    page_header(
        "Model Performance",
        "Comparison scores are read from models/model_comparison.csv. "
        "The confusion matrix is calculated from the saved model with the original test split.",
    )

    comparison = get_model_comparison()
    model_name = saved_model_name(model)

    if comparison is None:
        st.info("Run train_model.py to generate model comparison results.")
    else:
        st.markdown(f"**Saved model:** {model_name}")
        saved_row = comparison[comparison["Model"] == model_name]

        if len(saved_row) == 1:
            metric_columns = st.columns(4)
            for column, metric_name in zip(
                metric_columns,
                ["Accuracy", "Precision", "Recall", "F1 Score"],
            ):
                score = float(saved_row.iloc[0][metric_name])
                with column:
                    metric_card(
                        metric_name,
                        f"{score * 100:.1f}%",
                        f"Score {score:.3f}",
                        accent=metric_name == "F1 Score",
                    )
        else:
            st.info(
                "The saved model name was not found in model_comparison.csv, "
                "so the metric cards are hidden. The comparison table is still shown."
            )

        st.write("")
        chart_column, table_column = st.columns([1.3, 1])

        with chart_column:
            with st.container(border=True):
                st.markdown("**Model comparison**")
                melted = comparison.melt(
                    id_vars="Model",
                    var_name="Metric",
                    value_name="Score",
                )
                comparison_chart = px.bar(
                    melted,
                    x="Model",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    color_discrete_sequence=[
                        "#0B1F44",
                        "#1D4ED8",
                        "#60A5FA",
                        "#93C5FD",
                    ],
                )
                comparison_chart.update_yaxes(range=[0, 1])
                style_chart(comparison_chart, height=420)
                st.plotly_chart(comparison_chart, width="stretch")

        with table_column:
            with st.container(border=True):
                st.markdown("**Scores from training**")
                formatted = comparison.copy()
                for metric_name in ["Accuracy", "Precision", "Recall", "F1 Score"]:
                    formatted[metric_name] = formatted[metric_name].map(
                        lambda value: f"{float(value):.3f}"
                    )
                st.dataframe(formatted, hide_index=True, width="stretch")
                st.caption(
                    "These values come from train_model.py. "
                    "The saved file is the model with the highest F1 score."
                )

    st.write("")
    with st.container(border=True):
        st.markdown("**Confusion matrix**")
        matrix = compute_confusion_matrix(model)
        matrix_figure = px.imshow(
            matrix,
            text_auto=True,
            x=["Predicted No Dropout", "Predicted Dropout"],
            y=["Actual No Dropout", "Actual Dropout"],
            color_continuous_scale=["#E8F2FC", "#1D4ED8", "#0B1F44"],
            aspect="equal",
        )
        matrix_figure.update_layout(xaxis_title="", yaxis_title="")
        style_chart(matrix_figure, height=460)
        st.plotly_chart(matrix_figure, width="stretch")
        st.caption(
            "Calculated with test_size=0.2, random_state=42, and stratify, "
            "the same split used in evaluate_model.py."
        )


# -----------------------------------
# APP
# -----------------------------------

def main():
    model = get_model()
    page = render_sidebar()

    if page == "Dashboard":
        render_dashboard(model)
    elif page == "Student Risk Prediction":
        render_prediction(model)
    elif page == "Explainable AI":
        render_explainable_ai(model)
    elif page == "Student Records":
        render_records(model)
    elif page == "Advisor Panel":
        render_advisor()
    elif page == "Model Performance":
        render_performance(model)

    render_sidebar_status()


if __name__ == "__main__":
    main()
