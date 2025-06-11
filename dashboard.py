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

    # Load only model-relevant columns
    model_columns = [
        'GeneralHealth', 'AgeCategory', 'HighBloodPressure', 'BMI',
        'Highcholesterol', 'AlcoholDrinkers', 'Gender', 'RaceEthnicityCategory',
        'PhysicalActivities', 'DifficultyWalking', 'HouseholdIncome', 'HadDiabetes'
    ]
    df = pd.read_csv("df_final.csv")[model_columns]

    # Human-readable mappings
    mappings = {
        'GeneralHealth': {1: 'Poor', 2: 'Fair', 3: 'Good', 4: 'Very Good', 5: 'Excellent'},
        'HighBloodPressure': {0: 'No', 1: 'Yes'},
        'Highcholesterol': {0: 'No', 1: 'Yes'},
        'AlcoholDrinkers': {0: 'No', 1: 'Yes'},
        'Gender': {0: 'Female', 1: 'Male'},
        'RaceEthnicityCategory': {0: 'Non-Hispanic', 1: 'Hispanic'},
        'PhysicalActivities': {0: 'No', 1: 'Yes'},
        'DifficultyWalking': {0: 'No', 1: 'Yes'},
        'HouseholdIncome': {
            1: '<25k', 2: '25k-50k', 3: '50k-75k', 4: '75k-100k', 5: '100k+'
        },
        'HadDiabetes': {0: 'No', 1: 'Yes'}
    }

    df_display = df.copy()
    for col, map_dict in mappings.items():
        df_display[col] = df_display[col].map(map_dict)

    # SECTION 1 — Pygwalker
    st.subheader("🚀 Interactive Dashboard")
    with st.expander("Click to launch full data explorer", expanded=True):
        @st.cache_resource
        def get_pyg_renderer():
            return StreamlitRenderer(df_display, spec="./chart_meta_0.json", kernel_computation=True)
        renderer = get_pyg_renderer()
        renderer.explorer()

    # SECTION 2 — Visuals
    st.subheader("📈 Key Visual Summaries")

    # Distribution Plot
    st.markdown("#### 📌 Distribution Plot")
    selected_dist = st.selectbox("Select column for distribution", df.columns)

    fig1, ax1 = plt.subplots()
    col_data = df[selected_dist].dropna()

    if selected_dist == "AgeCategory":
        age_bins = [0, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 150]
        age_labels = ['<25', '25-29', '30-34', '35-39', '40-44', '45-49',
                      '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']
        age_grouped = pd.cut(col_data, bins=age_bins, labels=age_labels, right=True)
        counts = age_grouped.value_counts().sort_index()
        bars = ax1.bar(counts.index.astype(str), counts.values, color="skyblue")
        
        ax1.set_title("Age Group Distribution")
        ax1.set_xlabel("Age Group")
        ax1.set_ylabel("Count")
        
        # Rotate x-axis labels
        ax1.set_xticklabels(counts.index.astype(str), rotation=45, ha='right', fontsize=9)
        
        # Add value labels on bars
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, int(yval), ha='center', va='bottom', fontsize=8)
        
        fig1.tight_layout()

    elif selected_dist in mappings:
        counts = df_display[selected_dist].value_counts()
        bars = ax1.bar(counts.index, counts.values, color="skyblue")
        ax1.set_title(f"{selected_dist} Distribution")
        ax1.set_xlabel(selected_dist)
        ax1.set_ylabel("Count")
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, int(yval), ha='center', va='bottom')
    else:
        sns.histplot(col_data, bins=30, kde=True, ax=ax1, color="skyblue")
        ax1.set_title(f"Distribution of {selected_dist}")
        ax1.set_xlabel(selected_dist)
        ax1.set_ylabel("Density")

    st.pyplot(fig1)

    # Boxplot
    st.markdown("#### 📦 Boxplot")
    # Only plot boxplot for columns that exist
    boxplot_candidates = ['AgeCategory']
    existing_boxplot_cols = [col for col in boxplot_candidates if col in df.columns]
    
    # Boxplot Section – Only for AgeCategory
    st.markdown("#### 📦 Boxplot")
    if 'AgeCategory' in df.columns:
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=df['AgeCategory'], ax=ax2, color="lightgreen")
        ax2.set_title("Boxplot of AgeCategory")
        ax2.set_xlabel("Age (in years)")
        st.pyplot(fig2)
    else:
        st.warning("⚠️ 'AgeCategory' column not found in dataset.")


    # Correlation Matrix
    st.markdown("#### 🔗 Correlation Matrix")
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax3)
        ax3.set_title("Correlation Matrix")
        st.pyplot(fig3)
    else:
        st.info("No numeric columns available for correlation.")

    # SECTION 3 — Descriptive Summary
    st.subheader("📘 Descriptive Summary")
    view_mode = st.radio("Choose view mode:", ['View Summary', 'View Column Names', 'View Unique Values'])

    if view_mode == 'View Summary':
        st.dataframe(df.describe(include='all'))
    elif view_mode == 'View Column Names':
        st.write(df.columns.tolist())
    elif view_mode == 'View Unique Values':
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            st.markdown(f"**{col}** ({len(unique_vals)} unique): {unique_vals[:20]}")
            if len(unique_vals) > 20:
                st.caption("Showing only first 20 unique values")
                
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
    st.markdown("""
    👉 [View Full Slides](https://your-link.com)
    """)

    st.subheader("🙌 Acknowledgements")
    st.markdown("""
    - Supervisor: Dr. Nazean Binti Jomhari
    - Course: WQD7023
    """)

