import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import joblib
import requests

st.set_page_config(layout="wide")

# =====================================
# 1. Load Model & Scaler
# =====================================
MODEL_PATH = "logistic_regression_water_quality.pkl"
SCALER_PATH = "scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    model = None
    scaler = None
    st.error(f"Error loading model/scaler: {e}")

PREDICTION_MAP = {
    0: "Class I",
    1: "Class IIA/IIB",
    2: "Class III",
    3: "Class IV",
    4: "Class V"
}

def local_ml_predict(features):
    if (model is None) or (scaler is None):
        return None
    X = np.array(features, dtype=float).reshape(1, -1)
    X = np.nan_to_num(X, nan=0.0)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return int(y_pred[0])

# =====================================
# 2. Deep Seek API Integration
# =====================================
DEEP_SEEK_API_KEY = st.secrets["DEEP_SEEK_API_KEY"]
DEEP_SEEK_URL = "https://api.deepseek.com/v1/chat/completions"

def call_deepseek_api(user_prompt, data_summary=""):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEP_SEEK_API_KEY}"
    }
    system_message = (
        "You are a helpful assistant specializing in practical river water improvement strategies. "
        "Provide actionable insights rather than technical or statistical details."
    )
    user_message = (
        f"{user_prompt}\n\nData summary:\n{data_summary}\n\n"
        "Focus on real-world actions to enhance river water quality."
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "stream": False
    }
    try:
        response = requests.post(DEEP_SEEK_URL, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                return "No valid 'choices' found in response."
        else:
            return f"API Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Request failed: {e}"

# =====================================
# 3. Column Normalization
# =====================================
CANONICAL_NAMES = {
    "time": "Time",
    "phsensor": "pH Sensor",
    "orpsensor": "ORP Sensor",
    "tdssensor": "TDS Sensor",
    "nhsensor": "NH Sensor",
    "dosensor": "DO Sensor",
    "codsensor": "COD Sensor",
    "bodsensor": "BOD Sensor",
    "trsensor": "Turbidity",
    "ctsensor": "CT Sensor"
}

COLUMN_ALIAS_MAP = {
    "phsensor": ["phsensor", "ph_sensor", "phvaluesgtelom"],
    "orpsensor": ["orpsensor", "orp_sensor", "orpsgtelom"],
    "tdssensor": ["tdssensor", "tds_sensor", "tdssgtelom"],
    "nhsensor": ["nhsensor", "nh_sensor", "nhsensensor", "nh_sgtelom"],
    "dosensor": ["dosensor", "do_sensor", "dosgtelom"],
    "codsensor": ["codsensor", "cod_sensor", "codsgtelom"],
    "bodsensor": ["bodsensor", "bod_sensor", "bodsgtelom"],
    "trsensor": ["trsensor", "tr_sensor", "trsgtelom"],
    "ctsensor": ["ctsensor", "ct_sensor", "ctsgtelom"]
}

def normalize_columns(df):
    for col in df.columns:
        if col.strip().lower() == "time":
            df.rename(columns={col: "Time"}, inplace=True)
    new_columns = {}
    for col in df.columns:
        if col == "Time":
            continue
        normalized = re.sub(r"[\s\._]+", "", col.lower())
        found = False
        for canonical_key, aliases in COLUMN_ALIAS_MAP.items():
            if normalized in aliases:
                new_columns[col] = CANONICAL_NAMES.get(canonical_key, col)
                found = True
                break
        if not found:
            new_columns[col] = col
    df.rename(columns=new_columns, inplace=True)
    return df

# =====================================
# 4. Cleaning & Klasifikasi
# =====================================
def clean_outliers(df, valid_ranges):
    for col, (min_val, max_val) in valid_ranges.items():
        if col in df.columns:
            daily_mean = df[(df[col] >= min_val) & (df[col] <= max_val)].resample('D')[col].mean()
            def replace_outlier(row):
                val = row[col]
                if pd.notna(val) and (val < min_val or val > max_val):
                    return daily_mean.get(row.name.date(), np.nan)
                return val
            df[col] = df.apply(replace_outlier, axis=1)
    return df

def fill_missing_with_daily_mean(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        grouped = df.groupby(df.index.date)[col]
        daily_means = grouped.transform('mean')
        df[col] = df[col].fillna(daily_means)
    return df

def ml_classify_row(row):
    feature_cols = [
        "pH Sensor", "ORP Sensor", "TDS Sensor",
        "NH Sensor", "DO Sensor", "BOD Sensor", "COD Sensor"
    ]
    features = [row.get(col, 0) for col in feature_cols]
    pred = local_ml_predict(features)
    return PREDICTION_MAP.get(pred, "Unknown") if pred is not None else None

# =====================================
# 5. Rule-Based Source Detection
# =====================================
source_ranges = {
    "Oil Palm Plantation": {
        "pH Sensor": (6.0, 6.8),
        "ORP Sensor": (250, 350),
        "Turbidity": (10, 50),
        "TDS Sensor": (200, 400),
        "NH Sensor": (0.3, 1.0),
        "COD Sensor": (40, 80),
        "DO Sensor": (4.0, 6.5),
        "BOD Sensor": (10, 20)
    },
    "FGV Palm Oil Mill": {
        "pH Sensor": (4.0, 6.0),
        "ORP Sensor": (150, 250),
        "Turbidity": (100, 300),
        "TDS Sensor": (300, 600),
        "NH Sensor": (1.0, 2.5),
        "COD Sensor": (200, 500),
        "DO Sensor": (1.0, 3.0),
        "BOD Sensor": (100, 250)
    },
    "FGV Nursery (Kechau 6)": {
        "pH Sensor": (6.5, 7.5),
        "ORP Sensor": (200, 350),
        "Turbidity": (50, 150),
        "TDS Sensor": (400, 800),
        "NH Sensor": (2.0, 5.0),
        "COD Sensor": (50, 100),
        "DO Sensor": (3.5, 5.0),
        "BOD Sensor": (20, 40)
    }
}

def detect_pollution_source(row):
    scores = {}
    for source, param_ranges in source_ranges.items():
        match = 0
        for param, (low, high) in param_ranges.items():
            value = row.get(param, None)
            if pd.notna(value) and low <= value <= high:
                match += 1
        scores[source] = match
    best_match = max(scores, key=scores.get)
    return best_match if scores[best_match] > 0 else "Unknown"

# ============= STREAMLIT APP =============
def main():
    st.title("Water Quality Dashboard with Source Detection and Deep Seek")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "Unnamed: 10" in df.columns:
            df.drop("Unnamed: 10", axis=1, inplace=True)
        df = normalize_columns(df)

        st.subheader("Raw Data")
        st.dataframe(df)

        if "Time" not in df.columns:
            st.error("No 'Time' column found.")
            return

        df["Time"] = pd.to_datetime(df["Time"], errors='coerce')
        df.set_index("Time", inplace=True)

        valid_ranges = {
            "pH Sensor": (0, 14),
            "ORP Sensor": (-1000, 1000),
            "TDS Sensor": (0, 5000),
            "NH Sensor": (0, 50),
            "DO Sensor": (0, 20),
            "BOD Sensor": (0, 50),
            "COD Sensor": (0, 500),
            "Turbidity": (0, 500),
            "CT Sensor": (0, 100)
        }

        for col in valid_ranges:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df_cleaned = clean_outliers(df.copy(), valid_ranges)
        df_cleaned = fill_missing_with_daily_mean(df_cleaned)

        if df_cleaned.empty:
            st.warning("Data is empty after cleaning.")
            return

        st.subheader("Classification and Pollution Source Detection")
        if (model is None) or (scaler is None):
            st.error("Model or Scaler not loaded.")
        else:
            df_classified = df_cleaned.copy()
            df_classified["WATER_CLASS"] = df_classified.apply(ml_classify_row, axis=1)
            df_classified["POLLUTION_SOURCE"] = df_classified.apply(detect_pollution_source, axis=1)
            st.dataframe(df_classified)

            class_counts = df_classified["WATER_CLASS"].value_counts()
            st.write("**Count per Class:**")
            st.write(class_counts)
            st.markdown("""
<b>Water Classes and Uses based on National Water Quality Standard (NWQS) for Malaysia:</b><br>
- <b>Class I</b>: Conservation of natural environment, suitable for very sensitive aquatic species and drinking water supply with minimal treatment.<br>
- <b>Class IIA/IIB</b>: Suitable for body-contact recreational use, fisheries, and drinking water supply after conventional treatment.<br>
- <b>Class III</b>: Suitable for irrigation, livestock watering, and drinking water supply after extended treatment.<br>
- <b>Class IV</b>: Suitable for industrial water supply only after treatment.<br>
- <b>Class V</b>: Very polluted, not suitable for any use.
""", unsafe_allow_html=True)


        st.divider()
        st.subheader("📊 Visualizations")
        numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        tab1, tab2, tab3, tab4 = st.tabs(["Parameter Trend", "Histogram", "Scatter Plot", "Correlation Matrix"])

        with tab1:
            st.write("#### Parameter Trend Over Time")
            trend_options = ["All Parameters"] + numerical_cols
            selected_param = st.selectbox("Select Parameter to Show", trend_options)
            agg_choice = st.selectbox("Aggregation", ["All time", "Daily", "Weekly", "Monthly"])
            df_for_trend = df_cleaned.copy()
            if agg_choice == "Daily":
                df_for_trend = df_cleaned.resample('D').mean()
            elif agg_choice == "Weekly":
                df_for_trend = df_cleaned.resample('W').mean()
            elif agg_choice == "Monthly":
                df_for_trend = df_cleaned.resample('M').mean()

            if selected_param != "All Parameters":
                fig_single = px.line(df_for_trend, x=df_for_trend.index, y=selected_param,
                                     title=f"{agg_choice} Trend - {selected_param}")
                fig_single.update_layout(xaxis_title="Time", yaxis_title=selected_param)
                st.plotly_chart(fig_single, use_container_width=True)
            else:
                for col in numerical_cols:
                    fig = px.line(df_for_trend, x=df_for_trend.index, y=col,
                                  title=f"{agg_choice} Trend - {col}")
                    fig.update_layout(xaxis_title="Time", yaxis_title=col)
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.write("#### Histogram")
            if numerical_cols:
                col_to_hist = st.selectbox("Select column to show histogram", numerical_cols)
                df_cleaned["MonthPeriod"] = df_cleaned.index.to_period("M")
                df_cleaned["Month-Name"] = df_cleaned.index.strftime("%B %Y")

                month_display_map = {
                    p: n for p, n in zip(df_cleaned["MonthPeriod"], df_cleaned["Month-Name"])
                }
                sorted_periods = sorted(df_cleaned["MonthPeriod"].unique())
                sorted_month_names = [month_display_map[p] for p in sorted_periods]
                selected_display = st.selectbox("Select Month", sorted_month_names)

                reverse_map = {v: k for k, v in month_display_map.items()}
                selected_period = reverse_map[selected_display]

                df_filtered = df_cleaned[df_cleaned["MonthPeriod"] == selected_period]

                if df_filtered.empty:
                    st.warning("No data available for the selected month.")
                else:
                    if col_to_hist in valid_ranges:
                        range_x = valid_ranges[col_to_hist]
                    else:
                        min_val = df_filtered[col_to_hist].min()
                        max_val = df_filtered[col_to_hist].max()
                        range_x = [min_val, max_val]

                    fig_hist = px.histogram(
                        df_filtered, x=col_to_hist, nbins=30, range_x=range_x,
                        title=f"Distribution of {col_to_hist} - {selected_display}"
                    )
                    fig_hist.update_layout(xaxis_title=col_to_hist, yaxis_title="Frequency")
                    st.plotly_chart(fig_hist, use_container_width=True)

                df_cleaned.drop(["MonthPeriod", "Month-Name"], axis=1, inplace=True)
            else:
                st.warning("No numerical columns available for histogram.")
                
        with tab3:
            st.write("#### Scatter Plot")
            if len(numerical_cols) >= 2:
                x_axis = st.selectbox("Select X-Axis", numerical_cols, index=0)
                y_axis = st.selectbox("Select Y-Axis", numerical_cols, index=1)
                fig_scatter = px.scatter(df_cleaned, x=x_axis, y=y_axis,
                                         title=f"Relationship between {x_axis} and {y_axis}")
                fig_scatter.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("Need at least two numerical columns.")
    

        with tab4:
            st.write("#### Correlation Matrix")
            if len(numerical_cols) > 1:
                corr = df_cleaned[numerical_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Need at least two numerical columns.")

        st.divider()
        st.subheader("💬 Get Insight from Deep Seek")

        user_prompt = st.text_area("Enter your question or prompt for Deep Seek:")
        if st.button("Ask Deep Seek"):
            if not user_prompt:
                st.error("Please enter a prompt to ask.")
            else:
                class_counts_str = f"Class counts: {class_counts.to_dict()}" if 'class_counts' in locals() else ""
                answer = call_deepseek_api(user_prompt, data_summary=class_counts_str)
                if answer.startswith("API Error") or answer.startswith("Request failed"):
                    st.error(answer)
                else:
                    st.success("Deep Seek Response:")
                    st.write(answer)
    else:
        st.info("Please upload a CSV file.")

if __name__ == "__main__":
    main()
