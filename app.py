import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load model bundle
# ----------------------------
bundle = joblib.load("gbt_hospital.pkl")
model = bundle["model"]
train_cols = bundle["train_columns"]
le = bundle.get("label_encoder", None)

# ----------------------------
# SAME feature engineering logic as your notebook
# (keep this consistent!)
# ----------------------------
def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # If in your notebook these fields were already grouped into these strings,
    # then you can skip transforms.
    # But we keep this robust in case user provides raw numeric visitors/rooms.
    def combine_extra_rooms(v):
        if v in ["0", "1", "2"]:
            return "0-2 rooms"
        elif v == "3":
            return "3 rooms"
        elif v == "4":
            return "4 rooms"
        else:
            return "5 or more rooms"

    def combine_visitors(v):
        if v in ["0", "1", "2"]:
            return "0-2 visitors"
        elif v == "3":
            return "3 visitors"
        elif v == "4":
            return "4 visitors"
        else:
            return "5 or more visitors"

    def combine_ward_type(v):
        if v in ["P", "T", "U"]:
            return "Other Ward Types (P, T, U)"
        return v

    def combine_age(v):
        if v in ["0-10", "11-20"]:
            return "0-20"
        if v in ["81-90", "91-100"]:
            return "81-100"
        return v

    if "Available_Extra_Rooms_in_Hospital" in df.columns:
        df["Available_Extra_Rooms_in_Hospital"] = df["Available_Extra_Rooms_in_Hospital"].apply(combine_extra_rooms)

    if "Patient_Visitors" in df.columns:
        df["Patient_Visitors"] = df["Patient_Visitors"].apply(combine_visitors)

    if "Ward_Type" in df.columns:
        df["Ward_Type"] = df["Ward_Type"].apply(combine_ward_type)

    if "Age" in df.columns:
        df["Age"] = df["Age"].apply(combine_age)
    return df

def preprocess_for_model(raw_df: pd.DataFrame) -> pd.DataFrame:
    engineered = engineer_features(raw_df)
    encoded = pd.get_dummies(engineered, drop_first=True)
    encoded = encoded.reindex(columns=train_cols, fill_value=0)  # align to training
    return encoded

# ----------------------------
# UI CATEGORIES (as you listed)
# ----------------------------
HOSPITAL_OPTIONS = list(range(1, 33))  # you can extend to full 32 if you want
# Better: use 1..32 since you said categories are [1..32]
HOSPITAL_OPTIONS = list(range(1, 33))

HOSPITAL_TYPE_OPTIONS = [0, 1, 2, 3, 4, 5, 6]
HOSPITAL_CITY_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13]
HOSPITAL_REGION_OPTIONS = [0, 1, 2]

EXTRA_ROOMS_OPTIONS = ["0", "1", "2", "3", "4", "5 or more"]
DEPARTMENT_OPTIONS = ["radiotherapy", "anesthesia", "gynecology", "TB & Chest disease", "surgery"]
WARD_TYPE_OPTIONS = ["R", "S", "Q", "Other Ward Types (P, T, U)"]
WARD_FACILITY_OPTIONS = ["A", "B", "C", "D", "E", "F"]
BED_GRADE_OPTIONS = [1.0, 2.0, 3.0, 4.0]
ADMISSION_OPTIONS = ["Emergency", "Trauma", "Urgent"]
SEVERITY_OPTIONS = ["Extreme", "Moderate", "Minor"]
VISITORS_OPTIONS = ["0", "1", "2", "3", "4", "5 or more"]
AGE_OPTIONS = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"]


# ----------------------------
# Streamlit Layout
# ----------------------------
st.set_page_config(page_title="Hospital Stay Predictor", layout="centered")
st.title("🏥 Hospital Stay Length Prediction")
st.write("Enter the patient & hospital details, then click **Predict**.")

with st.form("prediction_form"):
    st.subheader("Hospital Info")
    Hospital = st.selectbox("Hospital", HOSPITAL_OPTIONS, index=0)
    Hospital_type = st.selectbox("Hospital_type", HOSPITAL_TYPE_OPTIONS, index=2)
    Hospital_city = st.selectbox("Hospital_city", HOSPITAL_CITY_OPTIONS, index=0)
    Hospital_region = st.selectbox("Hospital_region", HOSPITAL_REGION_OPTIONS, index=0)

    st.subheader("Facility & Admission")
    Available_Extra_Rooms_in_Hospital = st.selectbox("Available_Extra_Rooms_in_Hospital", EXTRA_ROOMS_OPTIONS, index=0)
    Ward_Facility = st.selectbox("Ward_Facility", WARD_FACILITY_OPTIONS, index=0)
    Bed_Grade = st.selectbox("Bed_Grade", BED_GRADE_OPTIONS, index=1)
    Type_of_Admission = st.selectbox("Type of Admission", ADMISSION_OPTIONS, index=0)

    st.subheader("Patient Info")
    Department = st.selectbox("Department", DEPARTMENT_OPTIONS, index=2)
    Ward_Type = st.selectbox("Ward_Type", WARD_TYPE_OPTIONS, index=0)
    Illness_Severity = st.selectbox("Illness_Severity", SEVERITY_OPTIONS, index=1)
    Patient_Visitors = st.selectbox("Patient_Visitors", VISITORS_OPTIONS, index=0)
    Age = st.selectbox("Age", AGE_OPTIONS, index=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build raw input row (column names MUST match what you trained on BEFORE get_dummies)
    raw_input = pd.DataFrame([{
        "Hospital": Hospital,
        "Hospital_type": Hospital_type,
        "Hospital_city": Hospital_city,
        "Hospital_region": Hospital_region,
        "Available_Extra_Rooms_in_Hospital": Available_Extra_Rooms_in_Hospital,
        "Department": Department,
        "Ward_Type": Ward_Type,
        "Ward_Facility": Ward_Facility,
        "Bed_Grade": float(Bed_Grade),
        "Type of Admission": Type_of_Admission,
        "Illness_Severity": Illness_Severity,
        "Patient_Visitors": Patient_Visitors,
        "Age": Age
    }])
    print(raw_input)
    print(raw_input.isna().sum())
    try:
        X_ready = preprocess_for_model(raw_input)
        pred = model.predict(X_ready)[0]

        # Decode label if you used LabelEncoder in training
        if le is not None:
            pred_label = pred
        else:
            # fallback label mapping (change to your exact class names)
            pred_label = "31 days or more" if int(pred) == 1 else "30 days or less"

        st.success(f"✅ Prediction: **{pred_label}**")

        with st.expander("Show debug details (optional)"):
            st.write("Raw input:")
            st.dataframe(raw_input)
            st.write("Model-ready encoded input (aligned to training columns):")
            st.dataframe(X_ready)

    except Exception as e:
        st.error("Prediction failed. This usually means your Streamlit input columns don't match training columns.")
        st.exception(e)