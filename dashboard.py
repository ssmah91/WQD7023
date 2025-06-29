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
import streamlit.components.v1 as components
import base64

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Diabetes Mellitus Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Pages", ["Home", "Data Info", "Prediction", "About Project"])

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
        
        /* Only change the "Navigation" title text color */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: black !important;
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

elif page == "Data Info":
    st.title("📊 Data Information & Exploration")

    # Load and filter relevant columns
    model_columns = [
        'GeneralHealth', 'AgeCategory', 'HighBloodPressure', 'BMI',
        'Highcholesterol', 'AlcoholDrinkers', 'Gender', 'RaceEthnicityCategory',
        'PhysicalActivities', 'DifficultyWalking', 'HouseholdIncome', 'HadDiabetes'
    ]
    df = pd.read_csv("df_final.csv")
    df = df[model_columns]

    # Map BMI into categories (Normal, Overweight, Obese)
    def categorize_bmi(bmi):
        if bmi == 1:
            return 'Normal'
        elif bmi == 2:
            return 'Overweight'
        else:
            return 'Obese'
    df['BMI'] = df['BMI'].apply(categorize_bmi)

    # Mapping for all other categorical values
    mappings = {
        'GeneralHealth': {1: 'Poor', 2: 'Fair', 3: 'Good', 4: 'Very Good', 5: 'Excellent'},
        'HighBloodPressure': {0: 'No', 1: 'Yes'},
        'Highcholesterol': {0: 'No', 1: 'Yes'},
        'AlcoholDrinkers': {0: 'No', 1: 'Yes'},
        'Gender': {0: 'Female', 1: 'Male'},
        'PhysicalActivities': {0: 'No', 1: 'Yes'},
        'DifficultyWalking': {0: 'No', 1: 'Yes'},
        'HouseholdIncome': {
            1: '<25k', 2: '25k-50k', 3: '50k-75k', 4: '75k-100k', 5: '100k+'
        },
        'HadDiabetes': {0: 'No', 1: 'Yes'},
        'RaceEthnicityCategory': {0: 'Non-Hispanic', 1: 'Hispanic'}
    }

    df_display = df.copy()
    for col, mapping in mappings.items():
        if col in df_display.columns:
            df_display[col] = df_display[col].map(mapping)

    # === DISTRIBUTION VISUALIZER ===
    st.subheader("📊 Distribution Visualizer")

    selected_col = st.selectbox("Select a feature to visualize:", df_display.columns)
    chart_type = st.radio("Choose chart type:", ['Bar Chart', 'Pie Chart', 'Horizontal Bar'])

    value_counts = df_display[selected_col].value_counts(dropna=False)
    percent = (value_counts / len(df_display)) * 100
    plot_df = pd.DataFrame({
        'Category': value_counts.index.astype(str),
        'Count': value_counts.values,
        'Percentage': percent.round(2)
    })

    if chart_type == 'Bar Chart':
        fig, ax = plt.subplots()
        bars = ax.bar(plot_df['Category'], plot_df['Count'], color='skyblue')
        ax.set_title(f"Distribution of {selected_col}")
        ax.set_ylabel("Count")
        ax.set_xlabel("Category")
        ax.tick_params(axis='x', rotation=45)
        for bar, pct in zip(bars, plot_df['Percentage']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{int(bar.get_height())} ({pct:.1f}%)", 
                    ha='center', va='bottom')
        st.pyplot(fig)

    elif chart_type == 'Pie Chart':
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            plot_df['Count'], labels=plot_df['Category'], autopct='%1.1f%%', startangle=90)
        ax.set_title(f"Distribution of {selected_col}")
        st.pyplot(fig)

    elif chart_type == 'Horizontal Bar':
        fig, ax = plt.subplots()
        bars = ax.barh(plot_df['Category'], plot_df['Count'], color='skyblue')
        ax.set_title(f"Distribution of {selected_col}")
        ax.set_xlabel("Count")
        for i, (count, pct) in enumerate(zip(plot_df['Count'], plot_df['Percentage'])):
            ax.text(count + 1, i, f"{count} ({pct:.1f}%)", va='center')
        st.pyplot(fig)

    # === DESCRIPTIVE SUMMARY ===
    st.subheader("📘 Descriptive Summary")
    view_mode = st.radio("Choose a view:", ['View Summary', 'View Column Names', 'View Unique Values'])

    if view_mode == 'View Summary':
        st.write(df.describe(include='all'))

    elif view_mode == 'View Column Names':
        st.write(df.columns.tolist())

    elif view_mode == 'View Unique Values':
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            st.markdown(f"**{col}** ({len(unique_vals)} unique): {unique_vals[:20]}")
            if len(unique_vals) > 20:
                st.caption("🔎 Showing only first 20 unique values")

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

    # === Mapping Functions ===
    def map_age_to_category(age):
        if age < 25: return '18-24'
        elif age < 30: return '25-29'
        elif age < 35: return '30-34'
        elif age < 40: return '35-39'
        elif age < 45: return '40-44'
        elif age < 50: return '45-49'
        elif age < 55: return '50-54'
        elif age < 60: return '55-59'
        elif age < 65: return '60-64'
        elif age < 70: return '65-69'
        elif age < 75: return '70-74'
        elif age < 80: return '75-79'
        else: return '80+'

    def map_income_to_category(income):
        if income < 25000: return '<25k'
        elif income < 50000: return '25k-50k'
        elif income < 75000: return '50k-75k'
        elif income < 100000: return '75k-100k'
        else: return '100k+'

    def categorize_bmi(bmi):
        if bmi < 25: return 'Normal'
        elif bmi < 30: return 'Overweight'
        else: return 'Obese'

    # ==============================
    # Option 1: Manual Input Form
    # ==============================
    with st.expander("🧾 Check My Diabetes Risk", expanded=False):
        with st.form("form_input"):
            general_health = st.selectbox('General Health', ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'])
            age = st.number_input('Age', min_value=18, max_value=120, value=30)
            high_bp = st.selectbox('High Blood Pressure', ['Yes', 'No'])
            bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
            high_cholesterol = st.selectbox('High Cholesterol', ['Yes', 'No'])
            alcohol = st.selectbox('Alcohol Drinkers', ['Yes', 'No'])
            gender = st.selectbox('Gender', ['Male', 'Female'])
            race_ethnicity = st.selectbox('Race/Ethnicity', ['White', 'Black', 'Asian', 'Hispanic', 'Other'])
            physical_activity = st.selectbox('Physical Activities', ['Yes', 'No'])
            walking_diff = st.selectbox('Difficulty Walking', ['Yes', 'No'])
            income = st.number_input('Household Income (numeric)', min_value=0, value=50000)
            submit = st.form_submit_button("Predict")

        if submit:
            input_data = pd.DataFrame([{
                'GeneralHealth': general_health,
                'AgeCategory': map_age_to_category(age),
                'HighBloodPressure': 1 if high_bp == 'Yes' else 0,
                'BMI': pd.Series(categorize_bmi(bmi)).map({'Normal': 1, 'Overweight': 2, 'Obese': 3})[0],
                'Highcholesterol': 1 if high_cholesterol == 'Yes' else 0,
                'AlcoholDrinkers': 1 if alcohol == 'Yes' else 0,
                'Gender': 1 if gender == 'Male' else 0,
                'RaceEthnicityCategory': 1 if race_ethnicity == 'Hispanic' else 0,
                'PhysicalActivities': 1 if physical_activity == 'Yes' else 0,
                'DifficultyWalking': 1 if walking_diff == 'Yes' else 0,
                'HouseholdIncome': map_income_to_category(income)
            }])

            input_data['GeneralHealth'] = input_data['GeneralHealth'].map({'Excellent': 5, 'Very Good': 4, 'Good': 3, 'Fair': 2, 'Poor': 1})
            input_data['AgeCategory'] = input_data['AgeCategory'].map({
                '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37,
                '40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57,
                '60-64': 62, '65-69': 67, '70-74': 72, '75-79': 77, '80+': 85
            })
            input_data['HouseholdIncome'] = input_data['HouseholdIncome'].map({
                '<25k': 1, '25k-50k': 2, '50k-75k': 3, '75k-100k': 4, '100k+': 5
            })

            input_data = input_data[feature_names]
            try:
                probability = model.predict_proba(input_data)[:, 1][0]
                prediction = int(probability >= threshold)
                st.success(f"Predicted Diabetes Risk: **{probability:.2%}**")
                if prediction == 1:
                    st.warning("⚠️ High Risk of Diabetes")
                    st.markdown("**System Feedback:** Please consult a healthcare provider and consider adopting a healthier lifestyle.")
                else:
                    st.info("🟢 Low Risk of Diabetes")
                    st.markdown("**System Feedback:** Your risk appears low. Maintain healthy habits and regular check-ups.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # ==============================
    # Option 2: Batch CSV Upload
    # ==============================
    with st.expander("📂 Batch Risk Prediction (CSV Upload)", expanded=False):
        st.markdown("#### 📄 CSV Format Information")
        st.markdown("""
        The uploaded CSV must include the following columns:

        - `GeneralHealth` (e.g., Excellent, Good, Poor)  
        - `Age` (numeric, e.g., 35)  
        - `HighBloodPressure`, `Highcholesterol`, `AlcoholDrinkers`, `PhysicalActivities`, `DifficultyWalking` (Yes/No)  
        - `Gender` (Male/Female)  
        - `RaceEthnicityCategory` (White, Black, Asian, Hispanic, Other)  
        - `BMI` (numeric)  
        - `HouseholdIncome` (numeric, e.g., 52000)

        ✅ Sample row:
        ```csv
        Excellent,30,Yes,28.5,Yes,No,Female,White,Yes,No,52000
        ```
        ❗ Make sure column names match **exactly**.
        """)

        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                required = ['GeneralHealth', 'Age', 'HighBloodPressure', 'BMI', 'Highcholesterol',
                            'AlcoholDrinkers', 'Gender', 'RaceEthnicityCategory',
                            'PhysicalActivities', 'DifficultyWalking', 'HouseholdIncome']
                if not set(required).issubset(df.columns):
                    st.error(f"Missing columns. Required: {required}")
                    st.stop()

                # Preprocessing
                df['AgeCategory'] = df['Age'].apply(map_age_to_category)
                df['IncomeCategory'] = df['HouseholdIncome'].apply(map_income_to_category)
                df['BMI'] = df['BMI'].apply(lambda x: 1 if x < 25 else (2 if x < 30 else 3))

                mappings = {
                    'GeneralHealth': {'Excellent': 5, 'Very Good': 4, 'Good': 3, 'Fair': 2, 'Poor': 1},
                    'AgeCategory': {'18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37,
                                    '40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57,
                                    '60-64': 62, '65-69': 67, '70-74': 72, '75-79': 77, '80+': 85},
                    'HighBloodPressure': {'Yes': 1, 'No': 0},
                    'Highcholesterol': {'Yes': 1, 'No': 0},
                    'AlcoholDrinkers': {'Yes': 1, 'No': 0},
                    'Gender': {'Male': 1, 'Female': 0},
                    'PhysicalActivities': {'Yes': 1, 'No': 0},
                    'DifficultyWalking': {'Yes': 1, 'No': 0},
                    'RaceEthnicityCategory': {'Hispanic': 1, 'White': 0, 'Black': 0, 'Asian': 0, 'Other': 0},
                    'IncomeCategory': {'<25k': 1, '25k-50k': 2, '50k-75k': 3, '75k-100k': 4, '100k+': 5}
                }

                for col, mapping in mappings.items():
                    if col in df.columns:
                        df[col] = df[col].map(mapping)

                df['HouseholdIncome'] = df['IncomeCategory']
                df_input = df[feature_names]

                probs = model.predict_proba(df_input)[:, 1]
                preds = (probs >= threshold).astype(int)

                df['DiabetesRiskProbability'] = probs
                df['Prediction'] = preds
                df['RiskLevel'] = df['Prediction'].map({0: 'Low Risk', 1: 'High Risk'})

                st.success("✅ Batch prediction completed.")
                st.dataframe(df[['DiabetesRiskProbability', 'RiskLevel']].join(df[feature_names]))

                st.download_button(
                    label="📥 Download Prediction Results",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name="diabetes_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Error during batch prediction: {e}")

elif page == "About Project":
    st.title("📚 About This Research Project")

    st.subheader("🎯 Research Overview")
    st.markdown("""
    This work, titled **"Predictive Modeling for Detection of Diabetes Mellitus: A Study on Risk Factors and Machine Learning Approaches,"**
    is part of my Master of Data Science research project at University of Malaya under the supervision of Dr. Nazean.
    It applies machine learning methods to identify individuals at risk for diabetes using health and lifestyle data from the BRFSS 2023 dataset.

    The model was built using Gradient Boosting and optimized with SVMSMOTE to address class imbalance.
    The app aims to support early identification and awareness.
    """)

    st.subheader("🛠️ Project Repository")
    st.markdown("""
    🔗 [View the code on GitHub](https://github.com/ssmah91/WQD7023)

    Contents:
    - Streamlit dashboard code
    - Pretrained models and preprocessing scripts
    - Dataset and analysis notebooks
    """)

    st.subheader("📽️ Research Slides")
    
    # # Optional external link (if hosted elsewhere too)
    # st.markdown("👉 [View Full Slides](https://your-link.com)")
    
    # Load your local PDF file
    with open("your_slide.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    
    # Embed the PDF using an iframe
    pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>
    """
    components.html(pdf_display, height=600)

    st.subheader("🙌 Acknowledgements")
    st.markdown("""
    - Supervisor: Dr. Nazean Binti Jomhari
    - Course: WQD7023
    """)

