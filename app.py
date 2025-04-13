import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Streamlit Page Config
st.set_page_config(page_title="Customer Churn Analysis", layout="wide")
st.title("📊 Customer Churn Analysis Dashboard")

# Sidebar for File Upload
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Process after Upload
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Data Quality Checks
    st.subheader("🔍 Data Quality Checks")
    st.write("### Missing Data:")
    st.write(df.isnull().sum())
    
    st.write("### Duplicated Rows:")
    st.write(df.duplicated().sum())

    st.write("### Summary Statistics:")
    st.write(df.describe())

    # Handle Missing Data: Use SimpleImputer to fill missing values
    imputer = SimpleImputer(strategy='mean')
    df_imputed = df.copy()
    df_imputed[df_imputed.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(
        df_imputed.select_dtypes(include=[np.number])
    )
    st.write("ℹ️ Missing data imputed (mean strategy for numeric features).")

    # Churn Distribution
    st.subheader("📉 Churn Distribution")
    if 'Churn' in df.columns:
        fig, ax = plt.subplots()
        df['Churn'].value_counts().plot.pie(
            autopct='%1.1f%%',
            startangle=90,
            labels=['No', 'Yes'],
            colors=['lightgreen', 'salmon'],
            ax=ax
        )
        ax.set_ylabel('')
        ax.set_title("Churn Breakdown")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Column 'Churn' not found in the dataset.")

    # Correlation Heatmap
    st.subheader("📌 Correlation Heatmap (Numeric Features)")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.info("ℹ️ No numeric features to show correlation.")

    # Model Building Section
    st.subheader("🧠 Build Logistic Regression Model")
    df_model = df.copy()
    df_model.dropna(inplace=True)

    # Encode Categorical Columns
    label_encoders = {}
    for col in df_model.select_dtypes(include=['object']).columns:
        if col != 'customerID':
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])
            label_encoders[col] = le

    # Features & Target
    if 'Churn' in df_model.columns:
        X = df_model.drop(['Churn', 'customerID'], axis=1, errors='ignore')
        y = df_model['Churn']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)

        # Hyperparameter Tuning using GridSearchCV
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['liblinear']
        }
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        st.write("✅ **Best Model (Logistic Regression)**")
        st.write(f"Best Hyperparameters: {grid_search.best_params_}")

        st.write("✅ **Classification Report**")
        st.code(classification_report(y_test, y_pred))

        st.write("🔍 **Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        st.pyplot(fig)

        # Alternative Models (Random Forest)
        st.subheader("🔄 Compare with Random Forest Model")
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)

        st.write("✅ **Random Forest Classification Report**")
        st.code(classification_report(y_test, rf_pred))

        st.write("🔍 **Random Forest Confusion Matrix**")
        rf_cm = confusion_matrix(y_test, rf_pred)
        fig, ax = plt.subplots()
        sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        st.pyplot(fig)

    else:
        st.error("❌ 'Churn' column not found. Model training skipped.")

    # Prediction from User Input
    st.subheader("🧾 Predict Churn (User Input)")
    input_data = {}

    for col in X.columns:
        if col in label_encoders:
            options = label_encoders[col].classes_.tolist()
            selected = st.selectbox(f"Select {col}", options)
            input_data[col] = label_encoders[col].transform([selected])[0]
        else:
            val = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))
            input_data[col] = val

    if st.button("🚀 Predict Churn"):
        user_df = pd.DataFrame([input_data])
        user_scaled = scaler.transform(user_df)
        prediction = best_model.predict(user_scaled)[0]
        result = "⚠️ Churn Likely" if prediction == 1 else "✅ Customer Likely to Stay"
        st.success(f"Prediction: **{result}**")

    # Download Data as Power BI (Excel format)
    st.subheader("📥 Download Transformed Data for Power BI")
    def convert_df_to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Churn Data', index=False)
            writer.close()
        return output.getvalue()

    churn_ready = df_model.copy()
    churn_ready['Churn_Predicted'] = best_model.predict(X_scaled)

    excel_data = convert_df_to_excel(churn_ready)

    st.download_button(
        label="📂 Download Excel for Power BI",
        data=excel_data,
        file_name='churn_analysis_data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
