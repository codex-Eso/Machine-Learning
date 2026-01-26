#import the necessary libraries first
import streamlit as st
import pandas as pd
import joblib

#load the model + the bundles we set in the ipynb
bundle = joblib.load("gbt_hospital.pkl")
model = bundle["model"]
train_cols = bundle["train_columns"]
le = bundle.get("label_encoder", None)

#we need to ensure whatever inputs we give in the web app is the correct data we are giving our model
def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    #for Available_Extra_Rooms_in_Hospital
    def combine_extra_rooms(v):
        if v in ["0", "1", "2"]:
            return "0-2 rooms"
        elif v == "3":
            return "3 rooms"
        elif v == "4":
            return "4 rooms"
        else:
            return "5 or more rooms"

    #for Patient_Visitors
    def combine_visitors(v):
        if v in ["0", "1", "2"]:
            return "0-2 visitors"
        elif v == "3":
            return "3 visitors"
        elif v == "4":
            return "4 visitors"
        else:
            return "5 or more visitors"

    #for Department
    def combine_departments(v):
        if v in ["TB & Chest disease", "surgery"]:
            return "TB & Chest disease + surgery"
        return v

    #for Ward_Type
    def combine_ward_type(v):
        if v in ["P", "T", "U"]:
            return "Other Ward Types (P, T, U)"
        return v

    #for Age
    def combine_age(v):
        if v >= 0 or v <= 20:
            return "0-20"
        elif v >= 21 or v <= 30:
            return "21-30"
        elif v >= 31 or v <= 40:
            return "31-40"
        elif v >= 41 or v <= 50:
            return "41-50"
        elif v >= 51 or v <= 60:
            return "51-60"
        elif v >= 61 or v <= 70:
            return "61-70"
        elif v >= 71 or v <= 80:
            return "71-80"
        if v >= 81 or v <= 100:
            return "81-100"

    #apply the changes for the respective columns into the dataframe
    if "Available_Extra_Rooms_in_Hospital" in df.columns:
        df["Available_Extra_Rooms_in_Hospital"] = df["Available_Extra_Rooms_in_Hospital"].apply(combine_extra_rooms)

    if "Patient_Visitors" in df.columns:
        df["Patient_Visitors"] = df["Patient_Visitors"].apply(combine_visitors)

    if "Department" in df.columns:
        df["Department"] = df["Department"].apply(combine_departments)

    if "Ward_Type" in df.columns:
        df["Ward_Type"] = df["Ward_Type"].apply(combine_ward_type)

    if "Age" in df.columns:
        df["Age"] = df["Age"].apply(combine_age)

    return df

#OHE the data we got before model prediction!!!
#need to do the same stuff we did in the ipynb file (OHE + feature engineering)
def preprocess_for_model(raw_df: pd.DataFrame) -> pd.DataFrame:
    engineered = engineer_features(raw_df).copy()#make sure the features are engineered properly for the model to predict

    #Start with all zeros in the exact shape the model expects
    X = pd.DataFrame(0, index=engineered.index, columns=train_cols)

    #OHE starts here!
    for col in engineered.columns:
        val = engineered.loc[engineered.index[0], col]

        #safety check -> if model expects the same train column, set it directly
        if col in X.columns:
            X.loc[engineered.index[0], col] = val
            continue

        #else name the column "<feature>_<value>" if it pass the safety check
        key = f"{col}_{val}"
        if key in X.columns:
            X.loc[engineered.index[0], key] = 1

    return X

#list down all the possible options (categories) for the each column/feature
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
AGE_OPTIONS = list(range(0, 101))

with st.form("prediction_form"):
    #title and description
    st.title("🏥 Hospital Stay Length Prediction")
    st.write("Enter the patient & hospital details, then click **Predict**.")

    #one part for the hospital info
    st.subheader("Hospital Info")
    Hospital = st.slider("Hospital (Hospital in numeric value)", min_value=min(HOSPITAL_OPTIONS), max_value=max(HOSPITAL_OPTIONS))
    Hospital_type = st.slider("Hospital Type (Hospital Type in numeric value)", min_value=min(HOSPITAL_TYPE_OPTIONS), max_value=max(HOSPITAL_TYPE_OPTIONS))
    Hospital_city = st.selectbox("Hospital City (Hospital City in numeric value)", HOSPITAL_CITY_OPTIONS)
    Hospital_region = st.selectbox("Hospital Region (Hospital Region in numeric value)", HOSPITAL_REGION_OPTIONS)

    #second part for the admission + facility info
    st.subheader("Facility & Admission")
    Available_Extra_Rooms_in_Hospital = st.selectbox("Number of Available Extra Rooms in Hospital", EXTRA_ROOMS_OPTIONS)
    Department = st.selectbox("Hospital Department", DEPARTMENT_OPTIONS)
    Ward_Type = st.selectbox("Ward Type", WARD_TYPE_OPTIONS)
    Ward_Facility = st.selectbox("Ward Facility", WARD_FACILITY_OPTIONS)
    Bed_Grade = st.selectbox("Bed Grade", BED_GRADE_OPTIONS)
    Type_of_Admission = st.selectbox("Type of Admission", ADMISSION_OPTIONS)

    #third part for patient info
    st.subheader("Patient Info")
    Illness_Severity = st.selectbox("Illness Severity", SEVERITY_OPTIONS)
    Patient_Visitors = st.selectbox("Number of Patient Visitors", VISITORS_OPTIONS)
    Age = st.slider("Age (0-100)", min_value=min(AGE_OPTIONS), max_value=max(AGE_OPTIONS))

    submitted = st.form_submit_button("Predict")

if submitted:
    #build the raw input rows we got from the frontend
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
    try:
        X_ready = preprocess_for_model(raw_input)
        pred = model.predict(X_ready)[0]
        #Decode label using LabelEncoder if le detected from model
        if le is not None:
            pred_label = pred
        else:
            #fallback label encoding of le for some reason didnt work
            pred_label = "31 days or more" if int(pred) == 1 else "30 days or less"
        st.success(f"✅ Prediction: **{pred_label}**")
    except Exception as e:
        st.error("Prediction failed. This usually means your Streamlit input columns don't match training columns.")
        st.exception(e)

#just note that our model is not perfect and it will have a hard time detecting
#31 days or more stay days due to a lower recall score in this category
#where recall score showed us how many actual long stay days cases the model correctly finds