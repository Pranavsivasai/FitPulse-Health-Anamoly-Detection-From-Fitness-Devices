# ==========================================
# ChatGPT - code
# ==========================================

# import streamlit as st
# import pandas as pd
# import numpy as np

# # ------------------------------------------
# # PAGE CONFIG
# # ------------------------------------------
# st.set_page_config(page_title="Fitness Data Processor",
#                    layout="wide",
#                    page_icon="📊")

# # ------------------------------------------
# # CUSTOM CSS (Gradient + Hover Effects)
# # ------------------------------------------
# st.markdown("""
# <style>
# body {
#     background: linear-gradient(to right, #1f4037, #99f2c8);
# }

# .stButton>button {
#     background: linear-gradient(to right, #4e54c8, #8f94fb);
#     color: white;
#     border-radius: 10px;
#     height: 3em;
#     width: 100%;
#     font-weight: bold;
#     transition: 0.3s;
# }

# .stButton>button:hover {
#     transform: scale(1.05);
#     background: linear-gradient(to right, #43cea2, #185a9d);
# }

# .block-container {
#     padding-top: 2rem;
# }
# </style>
# """, unsafe_allow_html=True)

# # ------------------------------------------
# # TITLE
# # ------------------------------------------
# st.title("📊 Fitness Health Data Preprocessing Dashboard")
# st.write("Upload your fitness dataset and perform professional preprocessing.")

# # ------------------------------------------
# # SESSION STATE INITIALIZATION
# # ------------------------------------------
# if "raw_df" not in st.session_state:
#     st.session_state.raw_df = None

# if "clean_df" not in st.session_state:
#     st.session_state.clean_df = None

# # ------------------------------------------
# # FILE UPLOAD
# # ------------------------------------------
# uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.session_state.raw_df = df
#     st.success("File Uploaded Successfully!")

# # ------------------------------------------
# # CHECK NULL VALUES BUTTON
# # ------------------------------------------
# if st.session_state.raw_df is not None:
    
#     if st.button("🔍 Check Null Values"):
#         null_values = st.session_state.raw_df.isnull().sum()
#         st.subheader("Null Values in Each Column")
#         st.dataframe(null_values)

# # ------------------------------------------
# # PREPROCESSING BUTTON
# # ------------------------------------------
# if st.session_state.raw_df is not None:

#     if st.button("⚙️ Preprocess Data"):
        
#         df = st.session_state.raw_df.copy()

#         st.subheader("Preprocessing Log")

#         # Convert Date
#         if "Date" in df.columns:
#             df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
#             st.write("✔ Date column converted to datetime format")

#         # Identify numeric columns
#         numeric_cols = [
#             "Hours_Slept",
#             "Water_Intake (Liters)",
#             "Active_Minutes",
#             "Heart_Rate (bpm)"
#         ]

#         # Show null values before processing
#         null_before = df[numeric_cols].isnull().sum()
#         st.write("Null values before preprocessing:")
#         st.dataframe(null_before)

#         # Interpolation
#         if "User_ID" in df.columns:
#             df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
#                 lambda x: x.interpolate(method="linear")
#             )

#             st.write(f"🔄 Interpolating missing values for columns: {', '.join(numeric_cols)}")

#             df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
#                 lambda x: x.ffill().bfill()
#             )

#             st.write("✔ Forward & Backward fill applied")

#         # Workout_Type handling
#         if "Workout_Type" in df.columns:
#             df["Workout_Type"] = df["Workout_Type"].fillna("No Workout")
#             st.write("✔ Filled missing Workout_Type with 'No Workout'")

#         # Store cleaned data
#         st.session_state.clean_df = df

#         st.success("Preprocessing Completed Successfully!")

# # ------------------------------------------
# # PREVIEW CLEANED DATA
# # ------------------------------------------
# if st.session_state.clean_df is not None:

#     if st.button("👁 Preview Cleaned Data"):
#         st.subheader("Cleaned Dataset Preview")
#         st.dataframe(st.session_state.clean_df.head())

#     if st.button("📊 Check Null Values After Cleaning"):
#         st.subheader("Null Values After Cleaning")
#         st.dataframe(st.session_state.clean_df.isnull().sum())

# # ------------------------------------------
# # FOOTER
# # ------------------------------------------
# st.markdown("---")
# st.markdown("Developed for Professional Health Data Analytics 🚀")













#===========================================
# Claude Code
#===========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fitness Data Pro",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  (gradient + hover animations)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
#  DARK / LIGHT MODE INIT
# ─────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

dark_mode = st.session_state.dark_mode

# ── Theme variables ──
if dark_mode:
    bg          = "linear-gradient(135deg, #0f0c29, #302b63, #24243e)"
    text_col    = "#e0e0e0"
    sidebar_bg  = "linear-gradient(180deg, #1a1a2e 0%, #16213e 100%)"
    sidebar_bdr = "rgba(99,179,237,0.2)"
    card_bg     = "rgba(255,255,255,0.05)"
    card_bdr    = "rgba(99,179,237,0.25)"
    upload_bg   = "rgba(255,255,255,0.04)"
    upload_bdr  = "rgba(99,179,237,0.5)"
    metric_bg   = "rgba(255,255,255,0.06)"
    metric_bdr  = "rgba(183,148,244,0.2)"
    hr_col      = "rgba(99,179,237,0.2)"
    plot_bg_global = "#1a1a2e"
else:
    bg          = "linear-gradient(135deg, #f0f4ff, #faf0ff, #ffffff)"
    text_col    = "#1a1a2e"
    sidebar_bg  = "linear-gradient(180deg, #e8eaf6 0%, #d1c4e9 100%)"
    sidebar_bdr = "rgba(102,126,234,0.3)"
    card_bg     = "rgba(0,0,0,0.03)"
    card_bdr    = "rgba(102,126,234,0.3)"
    upload_bg   = "rgba(0,0,0,0.02)"
    upload_bdr  = "rgba(102,126,234,0.4)"
    metric_bg   = "rgba(0,0,0,0.04)"
    metric_bdr  = "rgba(118,75,162,0.25)"
    hr_col      = "rgba(102,126,234,0.25)"
    plot_bg_global = "#f8f9ff"

st.markdown(f"""
<style>
/* ── Background ── */
.stApp {{
    background: {bg};
    color: {text_col};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {sidebar_bg};
    border-right: 1px solid {sidebar_bdr};
}}

/* ── Title / Headings ── */
h1, h2, h3, h4 {{
    background: linear-gradient(90deg, #63b3ed, #b794f4, #f687b3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
}}

/* ── Card / metric boxes ── */
.card {{
    background: {card_bg};
    border: 1px solid {card_bdr};
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}}
.card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(99,179,237,0.25);
}}

/* ── Null info pill ── */
.null-pill {{
    display: inline-block;
    background: linear-gradient(90deg, #e53e3e, #c05621);
    border-radius: 999px;
    padding: 3px 14px;
    font-size: 13px;
    font-weight: 700;
    color: white;
    margin: 3px 4px;
}}
.clean-pill {{
    display: inline-block;
    background: linear-gradient(90deg, #38a169, #276749);
    border-radius: 999px;
    padding: 3px 14px;
    font-size: 13px;
    font-weight: 700;
    color: white;
    margin: 3px 4px;
}}

/* ── Custom Streamlit buttons ── */
.stButton > button {{
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 10px 28px;
    font-weight: 700;
    font-size: 15px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4);
}}
.stButton > button:hover {{
    background: linear-gradient(90deg, #764ba2, #667eea);
    transform: scale(1.04);
    box-shadow: 0 8px 25px rgba(118,75,162,0.55);
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border-radius: 12px;
    overflow: hidden;
}}

/* ── Progress / status ── */
.stProgress > div > div {{ background: linear-gradient(90deg,#63b3ed,#b794f4); }}

/* ── Upload box ── */
[data-testid="stFileUploadDropzone"] {{
    background: {upload_bg} !important;
    border: 2px dashed {upload_bdr} !important;
    border-radius: 14px !important;
    transition: border-color 0.3s;
}}
[data-testid="stFileUploadDropzone"]:hover {{
    border-color: #b794f4 !important;
}}

/* ── Divider ── */
hr {{ border-color: {hr_col}; }}

/* ── Metric numbers ── */
[data-testid="metric-container"] {{
    background: {metric_bg};
    border-radius: 12px;
    padding: 12px;
    border: 1px solid {metric_bdr};
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
for key, default in [("raw_df", None), ("clean_df", None), ("preprocessing_done", False), 
                      ("show_null_check", False), ("show_preview", False), ("show_eda", False)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💪 Fitness Data Pro")
    st.markdown("---")
    # ── Theme Toggle ──
    new_dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if new_dark != st.session_state.dark_mode:
        st.session_state.dark_mode = new_dark
        st.rerun()
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("1. 📂 Upload CSV")
    st.markdown("2. 🔍 Check Null Values")
    st.markdown("3. ⚙️ Preprocess Data")
    st.markdown("4. 👁 Preview Cleaned Data")
    st.markdown("5. 📊 Run EDA")
    st.markdown("---")
    st.markdown("<small style='color:#b794f4'>Built for Fitness & Health Tracking Datasets</small>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("# 🏋️ Fitness Health Data — Pro Pipeline")
st.markdown("Upload your fitness tracking CSV and let the pipeline preprocess, clean, and explore your data.")
st.markdown("---")


# ─────────────────────────────────────────────
#  STEP 1 — UPLOAD
# ─────────────────────────────────────────────
st.markdown("## 📂 Step 1 · Upload Dataset")
uploaded = st.file_uploader("Drop your CSV file here", type=["csv"])

if uploaded:
    df_raw = pd.read_csv(uploaded)
    # Only reset if a NEW file is uploaded
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded.name:
        st.session_state.last_uploaded_file = uploaded.name
        st.session_state.raw_df = df_raw
        st.session_state.preprocessing_done = False
        st.session_state.clean_df = None
        st.session_state.show_null_check = False
        st.session_state.show_preview = False
        st.session_state.show_eda = False
        st.session_state.log_lines = []
        st.session_state.before_nulls = None
        st.session_state.after_nulls = None

    if st.session_state.raw_df is not None:
        _df = st.session_state.raw_df
        st.markdown(f"""
        <div class='card'>
            ✅ &nbsp;<strong>{st.session_state.last_uploaded_file}</strong> loaded successfully!
            &nbsp;&nbsp;|&nbsp;&nbsp; 🗂 <strong>{_df.shape[0]:,}</strong> rows &nbsp;×&nbsp; <strong>{_df.shape[1]}</strong> columns
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{_df.shape[0]:,}")
        c2.metric("Columns", _df.shape[1])
        c3.metric("Total Nulls", int(_df.isnull().sum().sum()))


# ─────────────────────────────────────────────
#  STEP 2 — NULL VALUE CHECK
# ─────────────────────────────────────────────
if st.session_state.raw_df is not None:
    st.markdown("---")
    st.markdown("## 🔍 Step 2 · Check Null Values")

    if st.button("🔍 Check Null Values"):
        st.session_state.show_null_check = True

    if st.session_state.show_null_check:
        df = st.session_state.raw_df
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]

        if null_cols.empty:
            st.success("🎉 No null values found in the dataset!")
        else:
            st.markdown("### Null Values Detected")
            pills_html = ""
            for col, cnt in null_cols.items():
                pct = cnt / len(df) * 100
                pills_html += f"<span class='null-pill'>⚠ {col}: {cnt} ({pct:.1f}%)</span>"

            st.markdown(f"<div class='card'>{pills_html}</div>", unsafe_allow_html=True)

            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 3), facecolor="none")
            ax.set_facecolor("none")
            colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(null_cols)))
            bars = ax.barh(null_cols.index, null_cols.values, color=colors)
            ax.set_xlabel("Null Count", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor((1, 1, 1, 0.1))
            ax.bar_label(bars, padding=4, color="white", fontsize=10)
            st.pyplot(fig, transparent=True)


# ─────────────────────────────────────────────
#  STEP 3 — PREPROCESS
# ─────────────────────────────────────────────
if st.session_state.raw_df is not None:
    st.markdown("---")
    st.markdown("## ⚙️ Step 3 · Preprocess Data")

    if st.button("⚙️ Run Preprocessing"):
        df = st.session_state.raw_df.copy()

        log_lines = []

        # ── Date Parsing ──────────────────────────────
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
            log_lines.append(("info", "📅 Parsed <b>Date</b> column to datetime."))

        # ── Identify numeric targets ───────────────────
        numeric_candidates = [
            "Hours_Slept", "Water_Intake (Liters)",
            "Active_Minutes", "Heart_Rate (bpm)",
            "Steps_Taken", "Calories_Burned",
            "Stress_Level (1-10)"
        ]
        numeric_cols = [c for c in numeric_candidates if c in df.columns]

        # ── Interpolation per user ─────────────────────
        null_before = df[numeric_cols].isnull().sum()
        null_before_cols = null_before[null_before > 0].index.tolist()

        if "User_ID" in df.columns and null_before_cols:
            df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
                lambda x: x.interpolate(method="linear")
            )
            df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
                lambda x: x.ffill().bfill()
            )
            cols_str = ", ".join([f"<b>{c}</b>" for c in null_before_cols])
            log_lines.append(("success", f"🔢 Interpolated (linear) + ffill/bfill null values in: {cols_str}"))
        elif null_before_cols:
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear").ffill().bfill()
            cols_str = ", ".join([f"<b>{c}</b>" for c in null_before_cols])
            log_lines.append(("success", f"🔢 Interpolated null values in: {cols_str}"))

        # ── Categorical fill ───────────────────────────
        cat_fills = {"Workout_Type": "No Workout"}
        for col, fill_val in cat_fills.items():
            if col in df.columns:
                n = df[col].isnull().sum()
                if n > 0:
                    df[col] = df[col].fillna(fill_val)
                    log_lines.append(("success", f"🏷 Filled <b>{n}</b> null(s) in <b>{col}</b> → '<i>{fill_val}</i>'"))

        # ── Any remaining nulls → median fill ─────────
        remaining_null_cols = [c for c in df.columns if df[c].isnull().sum() > 0 and df[c].dtype in [np.float64, np.int64]]
        for col in remaining_null_cols:
            n = df[col].isnull().sum()
            df[col] = df[col].fillna(df[col].median())
            log_lines.append(("warning", f"📊 Filled remaining <b>{n}</b> null(s) in <b>{col}</b> with column median"))

        # ── Save everything to session state ──────────
        st.session_state.clean_df = df
        st.session_state.preprocessing_done = True
        st.session_state.log_lines = log_lines
        st.session_state.before_nulls = st.session_state.raw_df.isnull().sum()
        st.session_state.after_nulls = df.isnull().sum()

    # ── Always render log if preprocessing was done ──
    if st.session_state.preprocessing_done and "log_lines" in st.session_state:
        color_map = {"info": "#63b3ed", "success": "#68d391", "warning": "#f6ad55"}
        icon_map  = {"info": "ℹ️",      "success": "✅",        "warning": "⚠️"}

        st.markdown("### Preprocessing Log")
        log_html = ""
        for ltype, msg in st.session_state.log_lines:
            log_html += f"""
            <div style='padding:10px 16px; margin:6px 0; border-left:4px solid {color_map[ltype]};
                        background:rgba(255,255,255,0.04); border-radius:8px; font-size:14px;'>
                {icon_map[ltype]} &nbsp;{msg}
            </div>"""
        st.markdown(f"<div class='card'>{log_html}</div>", unsafe_allow_html=True)

        # ── Null comparison ────────────────────────────
        st.markdown("### Null Value Comparison")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Before Preprocessing**")
            before_nulls = st.session_state.before_nulls
            null_df_before = before_nulls[before_nulls > 0].rename("Null Count")
            st.dataframe(null_df_before, width='stretch')
        with cc2:
            st.markdown("**After Preprocessing**")
            after_nulls = st.session_state.after_nulls
            nulls_left = after_nulls[after_nulls > 0]
            if nulls_left.empty:
                st.success("🎉 Zero nulls remaining!")
            else:
                st.dataframe(nulls_left.rename("Null Count"), width='stretch')

                # ── Time Normalization Log ─────────────────────
        st.markdown("### 🕐 Time Normalization Log")
        df_check = st.session_state.clean_df

        time_log_html = ""

        if "Date" in df_check.columns:
            total_rows = len(df_check)
            valid_dates = df_check["Date"].notna().sum()
            invalid_dates = total_rows - valid_dates

            time_log_html += f"""
            <div style='padding:10px 16px; margin:6px 0; border-left:4px solid #63b3ed;
                        background:rgba(255,255,255,0.04); border-radius:8px; font-size:14px;'>
                📅 &nbsp;<b>Date</b> column parsed → <b>{valid_dates:,}</b> valid timestamps,
                <b style='color:#fc8181'>{invalid_dates}</b> invalid/coerced to NaT
            </div>"""

            # Frequency detection
            if valid_dates > 1:
                sample = df_check["Date"].dropna().sort_values()
                if "User_ID" in df_check.columns:
                    sample = df_check[df_check["User_ID"] == df_check["User_ID"].iloc[0]]["Date"].dropna().sort_values()
                diffs = sample.diff().dropna()
                if len(diffs) > 0:
                    median_freq = diffs.median()
                    total_seconds = int(median_freq.total_seconds())
                    if total_seconds < 60:
                        freq_label = f"{total_seconds} seconds"
                    elif total_seconds < 3600:
                        freq_label = f"{total_seconds // 60} minute(s)"
                    elif total_seconds < 86400:
                        freq_label = f"{total_seconds // 3600} hour(s)"
                    else:
                        freq_label = f"{total_seconds // 86400} day(s)"

                    time_log_html += f"""
                    <div style='padding:10px 16px; margin:6px 0; border-left:4px solid #68d391;
                                background:rgba(255,255,255,0.04); border-radius:8px; font-size:14px;'>
                        📊 &nbsp;Detected data frequency: <b>~{freq_label}</b> per record
                        (median interval across sample user)
                    </div>"""

            # Date range
            date_min = df_check["Date"].min()
            date_max = df_check["Date"].max()
            time_log_html += f"""
            <div style='padding:10px 16px; margin:6px 0; border-left:4px solid #b794f4;
                        background:rgba(255,255,255,0.04); border-radius:8px; font-size:14px;'>
                🗓 &nbsp;Date range: <b>{date_min.date()}</b> → <b>{date_max.date()}</b>
                &nbsp;|&nbsp; Span: <b>{(date_max - date_min).days} days</b>
            </div>"""

            time_log_html += f"""
            <div style='padding:10px 16px; margin:6px 0; border-left:4px solid #f6ad55;
                        background:rgba(255,255,255,0.04); border-radius:8px; font-size:14px;'>
                ⚠️ &nbsp;Timezone: timestamps stored as <b>local/naive</b>.
                UTC normalization skipped — dataset does not carry timezone info.
            </div>"""
        else:
            time_log_html += f"""
            <div style='padding:10px 16px; margin:6px 0; border-left:4px solid #fc8181;
                        background:rgba(255,255,255,0.04); border-radius:8px; font-size:14px;'>
                ❌ &nbsp;No <b>Date</b> column found in dataset — time normalization skipped.
            </div>"""

        st.markdown(f"<div class='card'>{time_log_html}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  STEP 4 — PREVIEW CLEAN DATA
# ─────────────────────────────────────────────
if st.session_state.preprocessing_done and st.session_state.clean_df is not None:
    st.markdown("---")
    st.markdown("## 👁 Step 4 · Preview Cleaned Dataset")

    if st.button("👁 Preview Cleaned Data"):
        st.session_state.show_preview = True

    if st.session_state.get("show_preview", False):
        df_clean = st.session_state.clean_df
        st.dataframe(df_clean.head(50), width='stretch')
        csv_bytes = df_clean.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv_bytes,
            file_name="cleaned_fitness_data.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────
#  STEP 5 — EDA
# ─────────────────────────────────────────────
if st.session_state.preprocessing_done and st.session_state.clean_df is not None:
    st.markdown("---")
    st.markdown("## 📊 Step 5 · Exploratory Data Analysis")

    if st.button("📊 Run Full EDA"):
        st.session_state.show_eda = True

    if st.session_state.get("show_eda", False):
        df = st.session_state.clean_df

        numeric_cols = [c for c in [
            "Steps_Taken", "Calories_Burned", "Hours_Slept",
            "Active_Minutes", "Heart_Rate (bpm)", "Stress_Level (1-10)"
        ] if c in df.columns]

        plot_bg    = "#1a1a2e"
        text_color = "white"

        def style_ax(ax):
            ax.set_facecolor(plot_bg)
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_edgecolor((1, 1, 1, 0.15))

        palette = ["#63b3ed", "#b794f4", "#f687b3", "#68d391", "#f6ad55", "#fc8181"]

        # ── 1. Distributions ──────────────────────────
        st.markdown("### 📈 Distribution of Numeric Features")
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), facecolor=plot_bg)
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols[:6]):
            sns.histplot(df[col], kde=True, ax=axes[i], color=palette[i], alpha=0.8)
            axes[i].set_title(f"Distribution of {col}")
            style_ax(axes[i])
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close(fig)

        # ── 2. Boxplots ───────────────────────────────
        st.markdown("### 📦 Outlier Detection (Boxplots)")
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), facecolor=plot_bg)
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols[:6]):
            sns.boxplot(x=df[col], ax=axes[i], color=palette[i])
            axes[i].set_title(f"Boxplot of {col}")
            style_ax(axes[i])
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close(fig)

        # ── 3. Correlation Heatmap ────────────────────
        st.markdown("### 🔥 Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(20, 5), facecolor=plot_bg)
        ax.set_facecolor(plot_bg)
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax,
                    linewidths=0.5, linecolor="#1a1a2e", annot_kws={"color": "white"})
        ax.tick_params(colors=text_color)
        plt.title("Correlation Heatmap", color=text_color)
        st.pyplot(fig, transparent=True)
        plt.close(fig)

        # ── 4. Time-Series (first user) ───────────────
        if "Date" in df.columns and "User_ID" in df.columns and "Heart_Rate (bpm)" in df.columns:
            st.markdown("### 📅 Heart Rate Time-Series (Sample User)")
            user_id   = df["User_ID"].iloc[0]
            user_data = df[df["User_ID"] == user_id].sort_values("Date")
            fig, ax   = plt.subplots(figsize=(12, 4), facecolor=plot_bg)
            ax.set_facecolor(plot_bg)
            ax.plot(user_data["Date"], user_data["Heart_Rate (bpm)"], color="#f687b3", linewidth=2)
            ax.fill_between(user_data["Date"], user_data["Heart_Rate (bpm)"], alpha=0.2, color="#f687b3")
            ax.set_title(f"Heart Rate Trend — User {user_id}", color=text_color)
            ax.tick_params(colors=text_color, axis="both")
            plt.xticks(rotation=45)
            style_ax(ax)
            st.pyplot(fig, transparent=True)
            plt.close(fig)


        # ── 5. Workout Type Pie ───────────────────────
        if "Workout_Type" in df.columns:
            st.markdown("### 🥧 Workout Type Distribution")
            wc  = df["Workout_Type"].value_counts()
            fig, ax = plt.subplots(figsize=(3, 3), facecolor=plot_bg)
            ax.set_facecolor(plot_bg)
            wedge_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(wc)))
            wedges, texts, autotexts = ax.pie(
                wc, labels=wc.index, autopct='%1.1f%%',
                startangle=90, colors=wedge_colors,
                wedgeprops=dict(edgecolor='#1a1a2e', linewidth=2)
            )
            for t in texts + autotexts:
                t.set_color(text_color)
            ax.set_title("Workout Type Distribution", color=text_color)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(fig, transparent=True)
            plt.close(fig)

        # ── 6. User Averages ──────────────────────────
        if "User_ID" in df.columns:
            st.markdown("### 👥 User-Level Average Summary")
            user_summary = df.groupby("User_ID")[numeric_cols].mean().round(2)
            st.dataframe(user_summary.head(20), width='stretch')

        st.success("✅ EDA complete!")


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:rgba(255,255,255,0.3); font-size:13px; padding:10px 0'>
    💪 Fitness Data Pro &nbsp;|&nbsp; Built with Streamlit &nbsp;|&nbsp; Professional EDA Pipeline
</div>
""", unsafe_allow_html=True)