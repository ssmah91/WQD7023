# Install libraries
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore", message=".*st.experimental_user.*", category=DeprecationWarning)

# Define folder path
path = r"\\"

st.set_page_config(
    page_title="Diabetes Mellitus Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Pages", ["Home", "Data Info", "Prediction"])

# Apply full-screen responsive background to Home page only
if page == "Home":
    st.markdown(
        """
        <style>
        .stApp::before {
            content: "";
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.5); /* Adjust darkness */
        }
        .stApp {
            background: url('https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExemp0Z2Z2ZWs1eWFpcGF2NXVhMnJvaGprbXRldXEzbzA0MjFhYnFidSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/7VzgMsB6FLCilwS30v/giphy.gif') no-repeat center center fixed;
            background-size: cover;
            color: white;
        }
        .big-font {
            font-size: 40px !important;
            font-weight: bold;
            text-shadow: 2px 2px #000;
        }
        .sub-text {
            font-size: 22px !important;
            text-shadow: 1px 1px #000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Render content
if page == "Home":
    st.markdown('<p class="big-font">Welcome to the Diabetes Mellitus Prediction App 🩺</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="sub-text">
        This application uses machine learning to predict the risk of Diabetes Mellitus based on health data.
        It empowers individuals and healthcare professionals with early risk assessments.
        </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p class="sub-text">
        Navigate using the sidebar to explore the dataset, run predictions, and view insightful visualizations.
        </p>
        """,
        unsafe_allow_html=True
    )

# elif page == "Data Info":
#     import pandas as pd

#     st.title("📊 Data Info Page")
#     df = pd.read_csv(path + "df_final.csv")

#     # Expander for Sample Data
#     with st.expander("🔍 Click to View Sample Data (Top 10 Rows)", expanded=False):
#         st.dataframe(df.head(10))

#     st.subheader("Columns Descriptions:")

#     option = st.radio(
#         "Select what you want to explore",
#         ('View Summary', 'View Column Names', 'View Unique Values')
#     )

#     if option == 'View Summary':
#         st.write("### Dataset Summary:")
#         st.write(df.describe(include='all'))

#     elif option == 'View Column Names':
#         st.write("### Column Names:")
#         st.write(df.columns.tolist())

#     elif option == 'View Unique Values':
#         st.write("### Unique Values per Column:")
#         with st.expander("📋 Click to Expand Column-wise Unique Values", expanded=False):
#             for col in df.columns:
#                 unique_vals = df[col].dropna().unique()
#                 st.write(f"**{col}** ({len(unique_vals)} unique): {unique_vals[:20]}")
#                 if len(unique_vals) > 20:
#                     st.caption("🔎 Only showing first 20 unique values.")

# Visualization page

elif page == "Data Info":
    # Set page config
    # st.title("📊 Interactive Chart Builder")

    # Load dataset
    df = pd.read_csv(path + "df_final.csv")  # Replace with your actual path

    # Mapping dictionaries
    binary_map = {0: 'No', 1: 'Yes'}
    gender_map = {0: 'Female', 1: 'Male'}
    general_health_map = {1: 'Poor', 2: 'Fair', 3: 'Good', 4: 'Very Good', 5: 'Excellent'}
    age_map = {
        1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44', 6: '45-49',
        7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69', 11: '70-74', 12: '75-79', 13: '80+'
    }
    race_map = {1: 'White', 2: 'Black', 3: 'Asian', 4: 'Hispanic', 5: 'Other'}
    income_map = {1: '<25k', 2: '25k-50k', 3: '50k-75k', 4: '75k-100k', 5: '100k+'}

    # Apply human-readable labels
    human_readable_columns = {
        'Gender': gender_map,
        'GeneralHealth': general_health_map,
        'AgeCategory': age_map,
        'RaceEthnicityCategory': race_map,
        'HouseholdIncome': income_map,
        'HighBloodPressure': binary_map,
        'Highcholesterol': binary_map,
        'AlcoholDrinkers': binary_map,
        'PhysicalActivities': binary_map,
        'DifficultyWalking': binary_map,
        'HadHeartAttack': binary_map,
        'HadDiabetes': binary_map,
        'HadHeartDisease': binary_map,
        'HadStroke': binary_map,
        'HadAsthma': binary_map,
        'HadSkinCancer': binary_map,
        'HadCOPD': binary_map,
        'HadDepressiveDisorder': binary_map,
        'HadKidneyDisease': binary_map,
        'HadArthritis': binary_map,
        'SmokerStatus': binary_map,
        'ECigaretteUsage': binary_map,
        'HadOtherCancer': binary_map,
        'HadInsurance': binary_map,
    }

    df_display = df.copy()
    for col, mapping in human_readable_columns.items():
        if col in df_display.columns:
            df_display[col] = df_display[col].map(mapping)

    # Run embedded Pygwalker with DuckDB backend
    @st.cache_resource

    def get_pyg_renderer() -> "StreamlitRenderer":
        return StreamlitRenderer(
        df_display,
        spec="./chart_meta_0.json",     # Save chart state here manually in UI
        kernel_computation=True         # DuckDB backend for performance
    )

    renderer = get_pyg_renderer()
    
    renderer.explorer()

elif page == "Prediction":
    st.title("🩺 Diabetes Risk Prediction")

    try:
        loaded = joblib.load('model_GradientBoosting_SVMSMOTE_with_threshold.pkl')
        model = loaded['model']
        threshold = loaded['threshold']
        feature_names = model.feature_names_in_
    except Exception as e:
        st.error(f"🚨 Error loading model: {e}")
        st.stop()

    st.subheader("Please input the following health information:")

    # Final features input form (11 features only)
    general_health = st.selectbox('General Health', ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'])
    age_category = st.selectbox('Age Category', ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'])
    high_bp = st.selectbox('High Blood Pressure', ['Yes', 'No'])
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
    high_cholesterol = st.selectbox('High Cholesterol', ['Yes', 'No'])
    alcohol = st.selectbox('Alcohol Drinkers', ['Yes', 'No'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    race_ethnicity = st.selectbox('Race/Ethnicity', ['White', 'Black', 'Asian', 'Hispanic', 'Other'])
    physical_activity = st.selectbox('Physical Activities', ['Yes', 'No'])
    walking_diff = st.selectbox('Difficulty Walking', ['Yes', 'No'])
    income = st.selectbox('Household Income', ['<25k', '25k-50k', '50k-75k', '75k-100k', '100k+'])

    if st.button("Predict"):
        input_data = pd.DataFrame([{
            'GeneralHealth': general_health,
            'AgeCategory': age_category,
            'HighBloodPressure': 1 if high_bp == 'Yes' else 0,
            'BMI': bmi,
            'Highcholesterol': 1 if high_cholesterol == 'Yes' else 0,
            'AlcoholDrinkers': 1 if alcohol == 'Yes' else 0,
            'Gender': 1 if gender == 'Male' else 0,
            'RaceEthnicityCategory': 1 if race_ethnicity == 'Hispanic' else 0,
            'PhysicalActivities': 1 if physical_activity == 'Yes' else 0,
            'DifficultyWalking': 1 if walking_diff == 'Yes' else 0,
            'HouseholdIncome': income
        }])

        # Encode categorical fields
        input_data['GeneralHealth'] = input_data['GeneralHealth'].map({'Excellent': 5, 'Very Good': 4, 'Good': 3, 'Fair': 2, 'Poor': 1})
        input_data['AgeCategory'] = input_data['AgeCategory'].map({
            '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37, 
            '40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57, 
            '60-64': 62, '65-69': 67, '70-74': 72, '75-79': 77, 
            '80+': 85
        })
        input_data['HouseholdIncome'] = input_data['HouseholdIncome'].map({
            '<25k': 1, '25k-50k': 2, '50k-75k': 3, '75k-100k': 4, '100k+': 5
        })
      
        def categorize_bmi(bmi):
            if bmi < 25:
                return 'Normal'
            elif bmi < 30:
                return 'Overweight'
            else:
                return 'Obese'

        bmi_category = categorize_bmi(bmi)
        input_data['BMI'] = pd.Series(bmi_category).map({'Normal': 1, 'Overweight': 2, 'Obese': 3})


        # Ensure correct column order
        input_data = input_data[feature_names]

        try:
            probability = model.predict_proba(input_data)[:, 1][0]
            prediction = int(probability >= threshold)  # Use optimized threshold

            st.success(f"Predicted Diabetes Risk: **{probability:.2%}**")
            if prediction == 1:
                st.warning("⚠️ The model predicts a **High Risk of Diabetes**.")
            else:
                st.info("🟢 The model predicts a **Low Risk of Diabetes**.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

