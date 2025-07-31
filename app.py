import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import joblib
import requests

# ------------------------------------------------------------
# 1. Konfigurasi awal: layout menjadi wide
# ------------------------------------------------------------
st.set_page_config(layout="wide")

# ------------------------------------------------------------
# 2. Logo Header (Tunggal PNG, Dipusatkan & Lebar Mendekati Judul)
# ------------------------------------------------------------
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.write("")
with col2:
    st.image("logo.png", use_container_width=True)
with col3:
    st.write("")
st.markdown("<div style='margin-top: -15px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# 3. Judul Utama Aplikasi
# ------------------------------------------------------------
st.title("iSENS-AIR: Artificial Intelligence for River Water Quality Monitoring")

# ------------------------------------------------------------
# 4. Load Model & Scaler
# ------------------------------------------------------------
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
    if model is None or scaler is None:
        return None
    X = np.array(features, dtype=float).reshape(1, -1)
    X = np.nan_to_num(X, nan=0.0)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return int(y_pred[0])

# ------------------------------------------------------------
# 5. Deep Seek API Integration
# ------------------------------------------------------------
DEEP_SEEK_API_KEY = "sk-9d758570ebfd4ae28221c2dde357b6d2"
DEEP_SEEK_URL = "https://api.deepseek.com/v1/chat/completions"

def generate_data_summary(df):
    summary = []
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "WATER_CLASS" in df.columns:
        class_counts = df["WATER_CLASS"].value_counts().to_dict()
        summary.append(f"Water Class Distribution: {class_counts}")
    
    # Hitung trend 30 hari terakhir
    df_30d = df.last("30D")
    if not df_30d.empty:
        trend_info = []
        for col in numerical_cols:
            trend_data = df_30d[col].resample("D").mean()
            if trend_data.dropna().shape[0] >= 2:
                slope = np.polyfit(range(len(trend_data)), trend_data.fillna(0), 1)[0]
                direction = "increasing" if slope > 0 else "decreasing"
                trend_info.append(f"{col}: {direction} (slope: {slope:.4f})")
        summary.append("Parameter Trend (last 30 days):\n" + "\n".join(trend_info))
    
    return "\n\n".join(summary)


def call_deepseek_api(user_prompt, data_summary=""):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEP_SEEK_API_KEY}"
    }
    system_message = (
        "You are a helpful and practical assistant specialized in water quality improvement. "
        "Your goal is to provide actionable and concise insights to help local governments, environmentalists, and citizens take meaningful actions.\n\n"
        "Avoid overly technical language. Avoid repeating the prompt. "
        "When relevant, refer to specific water classes (Class I to Class V) and related water usage categories."
    )
    full_user = (
        f"Based on the following question, give recommendations related to real-world river water management:\n"
        f"{user_prompt}\n\n"
        f"Data Summary (from recent observations):\n"
        f"{data_summary}\n\n"
        f"Available Parameters: pH, ORP, TDS, NH₃, DO, BOD, COD, Turbidity, Conductivity.\n"
        f"Water Quality Classes (from ML model): Class I (best) to Class V (worst).\n"
        f"Please structure your answer clearly, with bullet points if needed, and avoid generic statements."
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": full_user}
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


# ------------------------------------------------------------
# 6. Column Normalization
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 7. Cleaning Functions
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 8. Predefined Prompts for Deep Seek (diletakkan di bawah Deep Seek section)
# ------------------------------------------------------------
faq_categories = {
    "📊 Trend and Behavior Insights": [
        "Summarize the recent 7-day trend for all parameters.",
        "Which parameters show an increasing trend over the past month?",
        "Highlight any abnormal changes or spikes in the last 24 hours.",
        "Compare today’s readings with monthly average."
    ],
    "⚠ Anomaly and Threshold Detection": [
        "List all parameters that exceeded threshold limits in the last 7 days.",
        "Which parameter had the highest deviation from normal range?",
        "What is the likely cause of recent pH fluctuations?"
    ],
    "🧠 Prediction and Early Warning": [
        "Predict possible pollution risk in the next 3 days based on current data.",
        "What is the risk level of water quality deterioration this week?"
    ],
    "🏭 Source Attribution": [
        "Based on recent data patterns, what is the probable pollution source?",
        "Are the parameter spikes consistent with industrial activity from oil palm mill or nursery?"
    ],
    "📈 Performance and Quality Summary": [
        "Provide a summary of water quality condition for this week.",
        "Is the water quality within acceptable environmental standards?",
        "Which parameter most influences overall water quality today?"
    ]
}

# ------------------------------------------------------------
# 9. Streamlit App Logic Utama: Upload, Visualisasi, dan Deep Seek
# ------------------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Baca dan normalisasi kolom
    df = pd.read_csv(uploaded_file)
    if "Unnamed: 10" in df.columns:
        df.drop("Unnamed: 10", axis=1, inplace=True)
    df = normalize_columns(df)

    # Pastikan kolom Time ada, ubah ke datetime, set index
    if "Time" not in df.columns:
        st.error("No 'Time' column found.")
    else:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df.set_index("Time", inplace=True)

        # Definisikan rentang valid untuk setiap sensor
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

        # Konversi kolom ke numerik
        for col in valid_ranges:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Cleaning data
        df_cleaned = clean_outliers(df.copy(), valid_ranges)
        df_cleaned = fill_missing_with_daily_mean(df_cleaned)

        if df_cleaned.empty:
            st.warning("Data is empty after cleaning.")
        else:
            # -------------------------
            # 9a. Visualizations Langsung
            # -------------------------
            st.subheader("📊 Visualizations")

            numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            tab1, tab2, tab3, tab4 = st.tabs([
                "Parameter Trend",
                "Histogram",
                "Scatter Plot",
                "Correlation Matrix"
            ])
        with tab1:
            st.write("#### Parameter Trend Over Time")

            trend_options = ["All Parameters"] + numerical_cols
            selected_param = st.selectbox("Select Parameter to Show", trend_options)
            agg_choice = st.selectbox("Aggregation", ["All time", "Daily", "Weekly", "Monthly"])

            df_for_trend = df_cleaned.copy()

            # Pastikan indeks waktu sudah benar
            if not isinstance(df_for_trend.index, pd.DatetimeIndex):
                try:
                    df_for_trend.index = pd.to_datetime(df_for_trend.index)
                except:
                    st.warning("Index is not datetime. Cannot plot trend.")
                    st.stop()

            # Resampling berdasarkan pilihan
            if agg_choice == "Daily":
                df_for_trend = df_for_trend.resample('D').mean()
            elif agg_choice == "Weekly":
                df_for_trend = df_for_trend.resample('W').mean()
            elif agg_choice == "Monthly":
                df_for_trend = df_for_trend.resample('M').mean()

            df_for_trend = df_for_trend.sort_index()  # Sort waktu

            if selected_param != "All Parameters":
                if selected_param in df_for_trend.columns and not df_for_trend[selected_param].dropna().empty:
                    fig_single = px.line(
                        df_for_trend,
                        x=df_for_trend.index,
                        y=selected_param,
                        title=f"{agg_choice} Trend - {selected_param}"
                    )
                    fig_single.update_layout(
                        xaxis_title="Time",
                        yaxis_title=selected_param,
                        height=400,
                        margin=dict(l=40, r=40, t=40, b=40),
                        xaxis=dict(rangeslider=dict(visible=True))
                    )
                    st.plotly_chart(fig_single, use_container_width=True)
                else:
                    st.warning(f"No valid data found for {selected_param}.")
            else:
                valid_count = 0
                for col in numerical_cols:
                    if col in df_for_trend.columns and not df_for_trend[col].dropna().empty:
                        fig = px.line(
                            df_for_trend,
                            x=df_for_trend.index,
                            y=col,
                            title=f"{agg_choice} Trend - {col}"
                        )
                        fig.update_layout(
                            xaxis_title="Time",
                            yaxis_title=col,
                            height=400,
                            margin=dict(l=40, r=40, t=40, b=40),
                            xaxis=dict(rangeslider=dict(visible=True))
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        valid_count += 1
                if valid_count == 0:
                    st.warning("No valid numerical data available for trend visualization.")

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
                            df_filtered,
                            x=col_to_hist,
                            nbins=30,
                            range_x=range_x,
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
                    fig_scatter = px.scatter(
                        df_cleaned,
                        x=x_axis,
                        y=y_axis,
                        title=f"Relationship between {x_axis} and {y_axis}"
                    )
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

            # -------------------------
            # 9b. Deep Seek Section with Inline FAQ Prompts
            # -------------------------
            st.divider()
            st.subheader("💬 Get Insight with DeepSeek AI Integration")
            st.markdown(
                "Choose a question from the dropdowns below, then click **Get Insight** to analyze your data."
            )

            chosen_category = st.selectbox("Choose a Category", list(faq_categories.keys()))
            chosen_prompt = st.selectbox("Choose a Prompt", faq_categories[chosen_category])

            if st.button("Get Insight"):
                df_for_summary = df_cleaned.copy()
                feature_cols = [
                    "pH Sensor", "ORP Sensor", "TDS Sensor",
                    "NH Sensor", "DO Sensor", "BOD Sensor", "COD Sensor"
                ]
                df_for_summary["WATER_CLASS"] = df_for_summary.apply(
                    lambda row: PREDICTION_MAP.get(
                        local_ml_predict([row.get(col, 0) for col in feature_cols]),
                        "Unknown"
                    ),
                    axis=1
                )
                data_summary = generate_data_summary(df_for_summary)
                answer = call_deepseek_api(chosen_prompt, data_summary=data_summary)
                if answer.startswith("API Error") or answer.startswith("Request failed"):
                    st.error(answer)
                else:
                    st.success("Deep Seek Response:")
                    st.write(answer)

            # -------------------------
            # 9c. Tambahkan Separator Sebelum Tabel
            # -------------------------
            st.markdown("<hr>", unsafe_allow_html=True)  # <-- Garis pemisah

            # ------------------------------------------------------
            # 10. Bagian Tabel Klasifikasi (Label Dibuat Bold dan Lebih Besar)
            # ------------------------------------------------------
            # Tampilkan judul tabel dengan ukuran font lebih besar dan bold
            st.markdown(
                "<span style='font-size:20px; font-weight:bold;'>Classification Table</span>",
                unsafe_allow_html=True
            )
            # Expander dengan teks "Click to show table"
            with st.expander("Click to show table", expanded=False):
                df_classified = df_cleaned.copy()
                feature_cols = [
                    "pH Sensor", "ORP Sensor", "TDS Sensor",
                    "NH Sensor", "DO Sensor", "BOD Sensor", "COD Sensor"
                ]
                df_classified["WATER_CLASS"] = df_classified.apply(
                    lambda row: PREDICTION_MAP.get(
                        local_ml_predict([row.get(col, 0) for col in feature_cols]),
                        "Unknown"
                    ),
                    axis=1
                )
                st.dataframe(
                    df_classified.drop(columns=[c for c in df_classified.columns if c == "POLLUTION_SOURCE"], errors="ignore"),
                    use_container_width=True
                )

                class_counts = df_classified["WATER_CLASS"].value_counts(dropna=True)
                st.write("**Count per Class:**")
                st.write(class_counts)

                st.markdown("""
                ---
                **Water Classes and Uses (Summary)**  
                - **Class I**: Conservation of environment, minimal treatment (Fishery I).  
                - **Class IIA/IIB**: Conventional treatment or recreational with body contact (Fishery II).  
                - **Class III**: Extensive treatment, common tolerant species (Fishery III).  
                - **Class IV**: Irrigation/livestock drinking water.  
                - **Class V**: Worst quality.
                """)
