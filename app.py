
import streamlit as st
import pandas as pd
import joblib

# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# =====================================
# 2. INITIALIZE SESSION STATE
# =====================================

if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

if "comparison_results" not in st.session_state:
    st.session_state["comparison_results"] = None

# =====================================
# 3. LOAD MODEL AND DATASET
# =====================================

# Load time-validated model and matching preprocessor
model = joblib.load("models/crop_yield_time_model.pkl")

preprocessor = joblib.load(
    "models/crop_yield_time_preprocessor.pkl"
)

df = pd.read_csv("dataset/yield_df.csv")

 
# =====================================
# 4. APPLICATION TITLE
# =====================================

st.title("🌾 Crop Yield Prediction System")

st.write(
    "Predict agricultural crop yield using climate "
    "and agricultural data."
)

st.success("Machine learning model loaded successfully!")

# =====================================
# 5. MODEL PERFORMANCE DASHBOARD
# =====================================

# Model Performance Dashboard

st.divider()
st.subheader("📊 Model Performance Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "R² Score",
        "95.47%"
    )

with col2:
    st.metric(
        "Mean Absolute Error",
        "10,336.35"
    )

with col3:
    st.metric(
        "Root Mean Squared Error",
        "20,232.04"
    )

st.info(
    "Performance metrics are based on time-based validation "
    "using training data from 1990–2008 and testing data from 2009–2013."
)

# =====================================
# 6. AGRICULTURAL DETAILS
# =====================================

st.divider()
st.subheader("🌍 Select Agricultural Details")

country = st.selectbox(
    "Select Country",
    sorted(df["Area"].unique())
)

st.divider()
st.subheader("🌱 Multi-Crop Comparison")

selected_crops = st.multiselect(
    "Select crops to compare",
    sorted(df["Item"].unique()),
    default=["Potatoes"]
)

crop = st.selectbox(
    "Select Crop",
    sorted(df["Item"].unique())
)

# =====================================
# 7. CLIMATE AND AGRICULTURAL INPUTS
# =====================================

st.subheader("🌦️ Climate and Agricultural Information")

year = st.number_input(
    "Year",
    min_value=1990,
    max_value=2030,
    value=2026,
    step=1
)

rainfall = st.number_input(
    "Average Rainfall (mm/year)",
    min_value=51.0,
    max_value=3240.0,
    value=1000.0
)

pesticides = st.number_input(
    "Pesticides Usage (tonnes)",
    min_value=0.04,
    max_value=367778.0,
    value=1000.0
)

temperature = st.number_input(
    "Average Temperature (°C)",
    min_value=1.3,
    max_value=30.65,
    value=25.0
)

# =====================================
# 8. CLIMATE ANALYSIS
# =====================================

st.divider()
st.subheader("🌤️ Climate Analysis")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "🌧️ Entered Rainfall",
        f"{rainfall:.2f} mm/year"
    )

with col2:
    st.metric(
        "🌡️ Entered Temperature",
        f"{temperature:.2f} °C"
    )

crop_data = df[df["Item"] == crop]

if not crop_data.empty:

    min_rainfall = crop_data[
        "average_rain_fall_mm_per_year"
    ].min()

    max_rainfall = crop_data[
        "average_rain_fall_mm_per_year"
    ].max()

    min_temp = crop_data["avg_temp"].min()
    max_temp = crop_data["avg_temp"].max()

    st.write("**Historical dataset ranges for selected crop:**")

    st.write(
        f"Rainfall: {min_rainfall:.2f} to "
        f"{max_rainfall:.2f} mm/year"
    )

    st.write(
        f"Temperature: {min_temp:.2f} to "
        f"{max_temp:.2f} °C"
    )

    if min_rainfall <= rainfall <= max_rainfall:
        st.success(
            "Rainfall is within the historical dataset range."
        )
    else:
        st.warning(
            "Rainfall is outside the historical dataset range."
        )

    if min_temp <= temperature <= max_temp:
        st.success(
            "Temperature is within the historical dataset range."
        )
    else:
        st.warning(
            "Temperature is outside the historical dataset range."
        )

# =====================================
# 9. MULTI-CROP COMPARISON
# =====================================

st.divider()
st.subheader("📊 Multi-Crop Comparison Prediction")

if st.button("📊 Compare Selected Crops"):

    if not selected_crops:

        st.warning("Please select at least one crop.")

    else:

        comparison_data = pd.DataFrame({
            "Area": [country] * len(selected_crops),
            "Item": selected_crops,
            "Year": [year] * len(selected_crops),
            "average_rain_fall_mm_per_year": [rainfall] * len(selected_crops),
            "pesticides_tonnes": [pesticides] * len(selected_crops),
            "avg_temp": [temperature] * len(selected_crops)
        })

        processed_data = preprocessor.transform(
            comparison_data
        )

        comparison_predictions = model.predict(
            processed_data
        )

        comparison_results = pd.DataFrame({
            "Crop": selected_crops,
            "Predicted Yield (hg/ha)": comparison_predictions
        })

        # Save comparison results in session state
        st.session_state["comparison_results"] = comparison_results

        st.success("Crop comparison completed successfully!")

# Display saved comparison results
if st.session_state["comparison_results"] is not None:

    comparison_results = st.session_state["comparison_results"]

    st.subheader("📈 Multi-Crop Comparison Results")

    st.dataframe(
        comparison_results,
        use_container_width=True
    )

    st.bar_chart(
        comparison_results,
        x="Crop",
        y="Predicted Yield (hg/ha)"
    )

    # Download comparison results
    comparison_csv = comparison_results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Crop Comparison",
        data=comparison_csv,
        file_name="crop_comparison_results.csv",
        mime="text/csv"
    )

# =====================================
# 10. CROP YIELD PREDICTION
# =====================================

st.divider()
st.subheader("🌾 Crop Yield Prediction")

if st.button("🔍 Predict Crop Yield"):

    input_data = pd.DataFrame({
        "Area": [country],
        "Item": [crop],
        "Year": [year],
        "average_rain_fall_mm_per_year": [rainfall],
        "pesticides_tonnes": [pesticides],
        "avg_temp": [temperature]
    })

    processed_input = preprocessor.transform(
        input_data
    )

    prediction = model.predict(
        processed_input
    )

    # Save prediction to history
    st.session_state["prediction_history"].append({
        "Country": country,
        "Crop": crop,
        "Year": int(year),
        "Rainfall": rainfall,
        "Pesticides": pesticides,
        "Temperature": temperature,
        "Predicted Yield (hg/ha)": round(
            float(prediction[0]), 2
        )
    })

    st.success(
        f"Predicted Crop Yield: {prediction[0]:,.2f} hg/ha"
    )

    st.success("Prediction saved to history!")

# =====================================
# 11. PREDICTION HISTORY
# =====================================

st.divider()
st.subheader("📜 Prediction History")

if st.session_state["prediction_history"]:

    history_df = pd.DataFrame(
        st.session_state["prediction_history"]
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    # =================================
    # 12. CROP YIELD COMPARISON CHART
    # =================================

    st.subheader("📈 Crop Yield Comparison")

    st.bar_chart(
        history_df,
        x="Crop",
        y="Predicted Yield (hg/ha)"
    )

    # =================================
    # 13. PREDICTION INSIGHTS
    # =================================

    st.divider()
    st.subheader("📊 Prediction Insights Dashboard")

    highest_yield = history_df.loc[
        history_df["Predicted Yield (hg/ha)"].idxmax()
    ]

    lowest_yield = history_df.loc[
        history_df["Predicted Yield (hg/ha)"].idxmin()
    ]

    average_yield = history_df[
        "Predicted Yield (hg/ha)"
    ].mean()

    total_predictions = len(history_df)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🏆 Highest Yield Crop",
            highest_yield["Crop"],
            f'{highest_yield["Predicted Yield (hg/ha)"]:,.2f} hg/ha'
        )

    with col2:
        st.metric(
            "📉 Lowest Yield Crop",
            lowest_yield["Crop"],
            f'{lowest_yield["Predicted Yield (hg/ha)"]:,.2f} hg/ha'
        )

    with col3:
        st.metric(
            "📊 Average Predicted Yield",
            f"{average_yield:,.2f} hg/ha"
        )

    st.metric(
        "🔢 Total Predictions",
        total_predictions
    )

    # =================================
    # 14. CROP RECOMMENDATION
    # =================================

    st.divider()
    st.subheader("🌱 Crop Recommendation System")

    recommended_crop = highest_yield

    st.success(
        f"Recommended Crop: {recommended_crop['Crop']}"
    )

    st.write(
        f"Predicted Yield: "
        f"{recommended_crop['Predicted Yield (hg/ha)']:,.2f} hg/ha"
    )

    st.info(
        "Recommendation is based on the highest predicted "
        "yield among the crops tested in your prediction history."
    )

    # =================================
    # 15. DOWNLOAD PREDICTION HISTORY
    # =================================

    st.divider()

    csv = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv,
        file_name="crop_prediction_history.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No prediction history available yet. "
        "Make your first prediction!"
    )
