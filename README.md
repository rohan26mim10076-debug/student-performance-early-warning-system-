# student-performance-early-warning-system-
a simple project which tells student about their performance 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Student Early-Warning System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Early-Warning System")
st.write(
    "Upload student data to identify students who may need "
    "academic intervention."
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "📊 Dashboard",
        "📁 Upload & Clean Data",
        "🤖 Train Model",
        "⚠️ Risk Prediction",
        "📄 Reports"
    ]
)

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "data" not in st.session_state:
    st.session_state.data = None

if "model" not in st.session_state:
    st.session_state.model = None

if "features" not in st.session_state:
    st.session_state.features = None

if "metrics" not in st.session_state:
    st.session_state.metrics = None

if "predictions" not in st.session_state:
    st.session_state.predictions = None


# ---------------------------------------------------------
# SAMPLE DATA
# ---------------------------------------------------------

def create_sample_data():

    np.random.seed(42)

    n = 300

    data = pd.DataFrame({

        "student_id":
            range(1001, 1001 + n),

        "attendance":
            np.random.randint(40, 101, n),

        "assignment_score":
            np.random.randint(30, 101, n),

        "midterm_score":
            np.random.randint(30, 101, n),

        "study_hours":
            np.random.randint(1, 15, n),

        "previous_grade":
            np.random.randint(35, 101, n),

        "absences":
            np.random.randint(0, 20, n),

        "late_assignments":
            np.random.randint(0, 10, n)
    })

    # Create target using academic indicators
    risk_score = (
        (100 - data["attendance"]) * 0.25
        + (100 - data["assignment_score"]) * 0.20
        + (100 - data["midterm_score"]) * 0.25
        + (100 - data["previous_grade"]) * 0.15
        + data["absences"] * 1.2
        + data["late_assignments"] * 1.5
        - data["study_hours"] * 1.0
    )

    threshold = risk_score.median()

    data["at_risk"] = (
        risk_score > threshold
    ).astype(int)

    return data


# ---------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------

def clean_data(df):

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert columns to numeric where possible
    for column in df.columns:

        if column != "student_id":

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Fill missing numerical values
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        df[column] = df[column].fillna(
            df[column].median()
        )

    return df


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------

if page == "📊 Dashboard":

    st.header("📊 Student Performance Dashboard")

    if st.session_state.data is None:

        st.info(
            "No dataset loaded. Go to 'Upload & Clean Data' "
            "or generate the sample dataset."
        )

        if st.button("Generate Sample Dataset"):

            data = create_sample_data()

            st.session_state.data = data

            st.success("Sample dataset generated!")

            st.rerun()

    else:

        df = st.session_state.data

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Students",
                len(df)
            )

        with col2:

            if "at_risk" in df.columns:

                risk_count = int(
                    df["at_risk"].sum()
                )

                st.metric(
                    "At-Risk Students",
                    risk_count
                )

            else:
                st.metric(
                    "At-Risk Students",
                    "N/A"
                )

        with col3:

            if "attendance" in df.columns:

                st.metric(
                    "Average Attendance",
                    f"{df['attendance'].mean():.1f}%"
                )

        with col4:

            if "midterm_score" in df.columns:

                st.metric(
                    "Average Midterm",
                    f"{df['midterm_score'].mean():.1f}"
                )

        st.divider()

        st.subheader("Student Data")

        st.dataframe(
            df,
            use_container_width=True
        )

        # Risk distribution

        if "at_risk" in df.columns:

            st.subheader("Risk Distribution")

            risk_counts = df["at_risk"].value_counts()

            fig, ax = plt.subplots()

            ax.bar(
                ["Low Risk", "At Risk"],
                [
                    risk_counts.get(0, 0),
                    risk_counts.get(1, 0)
                ]
            )

            ax.set_ylabel("Number of Students")

            st.pyplot(fig)

# ---------------------------------------------------------
# UPLOAD AND CLEAN
# ---------------------------------------------------------

elif page == "📁 Upload & Clean Data":

    st.header("📁 Upload & Clean Student Data")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.subheader("Raw Data")

            st.dataframe(
                df.head(20),
                use_container_width=True
            )

            st.write(
                "Rows:",
                df.shape[0],
                "Columns:",
                df.shape[1]
            )

            st.subheader("Missing Values")

            st.dataframe(
                df.isnull().sum()
            )

            if st.button("Clean Dataset"):

                df = clean_data(df)

                st.session_state.data = df

                st.success(
                    "Dataset cleaned successfully!"
                )

                st.dataframe(
                    df,
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Error reading file: {e}"
            )

    st.divider()

    st.subheader("Don't have a dataset?")

    if st.button("Generate Sample Dataset"):

        df = create_sample_data()

        st.session_state.data = df

        st.success(
            "Sample dataset created successfully!"
        )

        st.dataframe(
            df.head(20),
            use_container_width=True
        )


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

elif page == "🤖 Train Model":

    st.header("🤖 Risk Prediction Model")

    if st.session_state.data is None:

        st.warning(
            "Please upload or generate a dataset first."
        )

    else:

        df = st.session_state.data.copy()

        if "at_risk" not in df.columns:

            st.error(
                "Dataset must contain an 'at_risk' target column."
            )

        else:

            st.subheader(
                "Training Random Forest Classifier"
            )

            # Remove ID
            features = [
                column
                for column in df.columns
                if column not in
                ["student_id", "at_risk"]
            ]

            X = df[features]

            y = df["at_risk"]

            # Ensure numeric data
            X = X.select_dtypes(
                include=np.number
            )

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )

            model = RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                max_depth=8
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            # Metrics
            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                zero_division=0
            )

            st.session_state.model = model

            st.session_state.features = list(
                X.columns
            )

            st.session_state.metrics = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            }

            # Display metrics

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Accuracy",
                    f"{accuracy:.2%}"
                )

            with col2:
                st.metric(
                    "Precision",
                    f"{precision:.2%}"
                )

            with col3:
                st.metric(
                    "Recall",
                    f"{recall:.2%}"
                )

            with col4:
                st.metric(
                    "F1 Score",
                    f"{f1:.2%}"
                )

            # Confusion matrix

            st.subheader(
                "Confusion Matrix"
            )

            cm = confusion_matrix(
                y_test,
                predictions
            )

            fig, ax = plt.subplots()

            ax.imshow(cm)

            ax.set_xlabel(
                "Predicted"
            )

            ax.set_ylabel(
                "Actual"
            )

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])

            for i in range(2):

                for j in range(2):

                    ax.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center"
                    )

            st.pyplot(fig)

            # Feature importance

            st.subheader(
                "Feature Importance"
            )

            importance = pd.DataFrame({

                "Feature":
                    X.columns,

                "Importance":
                    model.feature_importances_

            }).sort_values(
                "Importance",
                ascending=False
            )

            st.dataframe(
                importance,
                use_container_width=True
            )


# ---------------------------------------------------------
# RISK PREDICTION
# ---------------------------------------------------------

elif page == "⚠️ Risk Prediction":

    st.header("⚠️ Student Risk Prediction")

    if st.session_state.model is None:

        st.warning(
            "Please train the model first."
        )

    else:

        model = st.session_state.model

        features = st.session_state.features

        st.write(
            "Enter student information:"
        )

        values = {}

        columns = st.columns(2)

        for index, feature in enumerate(features):

            with columns[index % 2]:

                if feature == "attendance":

                    values[feature] = st.slider(
                        "Attendance (%)",
                        0,
                        100,
                        75
                    )

                elif feature in [
                    "assignment_score",
                    "midterm_score",
                    "previous_grade"
                ]:

                    values[feature] = st.slider(
                        feature.replace(
                            "_",
                            " "
                        ).title(),
                        0,
                        100,
                        60
                    )

                elif feature in [
                    "study_hours"
                ]:

                    values[feature] = st.number_input(
                        "Study Hours",
                        min_value=0,
                        max_value=24,
                        value=5
                    )

                elif feature in [
                    "absences",
                    "late_assignments"
                ]:

                    values[feature] = st.number_input(
                        feature.replace(
                            "_",
                            " "
                        ).title(),
                        min_value=0,
                        max_value=50,
                        value=2
                    )

                else:

                    values[feature] = st.number_input(
                        feature,
                        value=0.0
                    )

        if st.button(
            "Predict Student Risk"
        ):

            input_data = pd.DataFrame(
                [values]
            )

            prediction = model.predict(
                input_data
            )[0]

            probability = model.predict_proba(
                input_data
            )[0][1]

            if prediction == 1:

                st.error(
                    "⚠️ HIGH RISK: Student may need intervention."
                )

                st.write(
                    f"Risk probability: "
                    f"{probability:.2%}"
                )

                st.subheader(
                    "Recommended Intervention"
                )

                st.write(
                    """
                    • Contact the student  
                    • Monitor attendance  
                    • Provide academic support  
                    • Review assignment performance  
                    • Consider mentoring/tutoring  
                    """
                )

            else:

                st.success(
                    "✅ LOW RISK: Student currently appears to be performing adequately."
                )

                st.write(
                    f"Risk probability: "
                    f"{probability:.2%}"
                )


# ---------------------------------------------------------
# REPORTS
# ---------------------------------------------------------

elif page == "📄 Reports":

    st.header("📄 Reports & Export")

    if st.session_state.data is None:

        st.warning(
            "No student data available."
        )

    else:

        df = st.session_state.data.copy()

        # If model exists, predict all students

        if st.session_state.model is not None:

            model = st.session_state.model

            features = st.session_state.features

            X = df[features]

            predictions = model.predict(X)

            probabilities = model.predict_proba(X)[:, 1]

            report = df.copy()

            report["predicted_risk"] = predictions

            report["risk_probability"] = probabilities

            report["risk_level"] = np.where(
                predictions == 1,
                "HIGH RISK",
                "LOW RISK"
            )

            st.session_state.predictions = report

        else:

            report = df

        st.subheader(
            "Student Risk Report"
        )

        st.dataframe(
            report,
            use_container_width=True
        )

        # CSV export

        csv = report.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download CSV Report",
            data=csv,
            file_name="student_risk_report.csv",
            mime="text/csv"
        )

        # At-risk students

        if "predicted_risk" in report.columns:

            at_risk = report[
                report["predicted_risk"] == 1
            ]

            st.subheader(
                "⚠️ Students Requiring Attention"
            )

            st.dataframe(
                at_risk,
                use_container_width=True
            )

            st.write(
                f"Total students requiring attention: "
                f"**{len(at_risk)}**"
            )