from __future__ import annotations

import io
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

from utils.prediction import explain_prediction, predict_row
from utils.preprocessing import DATASET_URL, TARGET, build_features, clean_dataframe, load_dataset
from utils.training import train_models

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
DATASET_PATH = DATASET_DIR / "energy_dataset.csv"
MODEL_PATH = MODEL_DIR / "energy_prediction_model.pkl"

st.set_page_config(page_title="EcoMind AI", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# Professional light/dark modes are applied without changing the ML logic.
def inject_theme(dark: bool) -> None:
    if dark:
        bg, panel, text, muted, border, accent = "#081019", "#111d2a", "#eef2f5", "#91a0af", "rgba(255,255,255,.10)", "#8fe3bd"
    else:
        bg, panel, text, muted, border, accent = "#f4f7f8", "#ffffff", "#12202b", "#647382", "rgba(18,32,43,.12)", "#0a8f69"
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;700;800&family=Source+Sans+3:wght@400;500;600;700&display=swap');
      :root {{ --bg:{bg}; --panel:{panel}; --text:{text}; --muted:{muted}; --border:{border}; --accent:{accent}; }}
      .stApp {{ background: radial-gradient(circle at 78% 0%, rgba(68,151,129,.12), transparent 31%), var(--bg); color:var(--text); font-family:'Source Sans 3', sans-serif; }}
      h1,h2,h3,h4 {{ font-family:'Manrope', sans-serif !important; letter-spacing:-.045em; }}
      [data-testid="stSidebar"] {{ background: color-mix(in srgb, var(--bg) 90%, #12302d); border-right:1px solid var(--border); }}
      [data-testid="stMetric"] {{ background:var(--panel); border:1px solid var(--border); border-radius:16px; padding:16px; }}
      .glass {{ background:linear-gradient(135deg, color-mix(in srgb, var(--panel) 92%, transparent), color-mix(in srgb, var(--panel) 75%, transparent)); border:1px solid var(--border); border-radius:18px; padding:20px; box-shadow:0 18px 50px rgba(0,0,0,.08); }}
      .eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }}
      .hero-title {{ font-family:'Manrope',sans-serif; font-size:clamp(2.2rem, 5vw, 4.9rem); line-height:.98; letter-spacing:-.08em; margin:.6rem 0 1rem; }}
      .hero-title em {{ color:var(--accent); font-style:normal; }}
      .terminal {{ background:#07100f; color:#9ee7c4; border-radius:14px; padding:16px 18px; font-family:'DM Mono', monospace; font-size:.78rem; line-height:1.75; border:1px solid rgba(143,227,189,.20); box-shadow:inset 0 0 30px rgba(143,227,189,.03); }}
      .status-on {{ color:var(--accent); font-weight:700; }}
      .status-off {{ color:#dc7184; font-weight:700; }}
      .small-muted {{ color:var(--muted); font-size:.86rem; }}
      .step {{ display:flex; gap:12px; align-items:flex-start; padding:12px 0; border-bottom:1px solid var(--border); }}
      .step:last-child {{ border-bottom:0; }}
      .step-number {{ width:28px; height:28px; border-radius:9px; display:grid; place-items:center; background:color-mix(in srgb, var(--accent) 16%, transparent); color:var(--accent); font-weight:700; flex:0 0 auto; }}
      .stButton>button, .stDownloadButton>button {{ border-radius:10px; border:1px solid var(--border); }}
    </style>
    """, unsafe_allow_html=True)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "df" not in st.session_state:
    st.session_state.df = None
if "results" not in st.session_state:
    st.session_state.results = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "logs" not in st.session_state:
    st.session_state.logs = ["system :: EcoMind AI runtime ready", "source :: awaiting dataset handshake"]

inject_theme(st.session_state.dark_mode)

with st.sidebar:
    st.markdown("<div class='eyebrow'>EcoMind AI</div><h2 style='margin-top:6px'>Smart building<br>intelligence</h2>", unsafe_allow_html=True)
    st.caption("Production-style ML lifecycle workspace")
    st.session_state.dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode)
    st.divider()
    page = st.radio("Navigate", ["Overview", "Dataset & Upload", "EDA Studio", "Cleaning", "Preprocessing", "Model Lab", "Live Prediction", "Feature Importance", "Report"], label_visibility="collapsed")
    st.divider()
    dataset_ready = DATASET_PATH.exists() or st.session_state.df is not None
    model_ready = MODEL_PATH.exists() or st.session_state.results is not None
    st.markdown(f"**Connection status**")
    st.markdown(f"Dataset reader: <span class='{'status-on' if dataset_ready else 'status-off'}'>{'CONNECTED' if dataset_ready else 'WAITING'}</span>", unsafe_allow_html=True)
    st.markdown(f"Model runtime: <span class='{'status-on' if model_ready else 'status-off'}'>{'READY' if model_ready else 'NOT TRAINED'}</span>", unsafe_allow_html=True)
    st.markdown("Report engine: <span class='status-on'>AVAILABLE</span>", unsafe_allow_html=True)
    st.caption("Run locally with: streamlit run app.py")


def add_log(message: str) -> None:
    st.session_state.logs.append(message)
    st.session_state.logs = st.session_state.logs[-12:]


def get_active_df() -> pd.DataFrame:
    if st.session_state.df is None:
        st.session_state.df = load_dataset(DATASET_PATH)
        add_log(f"dataset :: loaded {len(st.session_state.df):,} rows from {DATASET_PATH}")
    return st.session_state.df


def render_terminal() -> None:
    lines = "\n".join(f"[{i + 1:02d}] {line}" for i, line in enumerate(st.session_state.logs[-8:]))
    st.markdown(f"<div class='terminal'>{lines}<br><span class='status-on'>▌ process stream active</span></div>", unsafe_allow_html=True)


def make_report(df: pd.DataFrame, results: dict, prediction: dict | None) -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27)); fig.patch.set_facecolor("#081019")
        fig.text(.08, .82, "EcoMind AI", color="#8fe3bd", fontsize=34, weight="bold")
        fig.text(.08, .72, "Smart Building Energy Consumption Prediction", color="white", fontsize=19, weight="bold")
        fig.text(.08, .62, f"Dataset: {len(df):,} rows · Target: {TARGET} · Model: Random Forest Regressor", color="#b5c2cf", fontsize=12)
        fig.text(.08, .47, "This report documents the complete ML lifecycle: source data, cleaning, preprocessing, model evaluation, prediction and explainability.", color="#b5c2cf", fontsize=14, wrap=True)
        pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27)); fig.suptitle("EDA snapshots", fontsize=20, weight="bold")
        axes[0].hist(df[TARGET].dropna(), bins=30, color="#8fe3bd", alpha=.88); axes[0].set_title("Target distribution"); axes[0].set_xlabel("Energy consumption (Wh)")
        corr = df.select_dtypes("number").corr()[TARGET].abs().sort_values(ascending=False).head(8).drop(TARGET, errors="ignore")
        axes[1].barh(corr.index[::-1], corr.values[::-1], color="#78b7ff"); axes[1].set_title("Absolute correlation with target"); axes[1].set_xlim(0, 1)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.69, 8.27)); fig.suptitle("Feature importance", fontsize=20, weight="bold")
        importance = results["feature_importance"].head(12).sort_values("importance")
        ax.barh(importance["feature"], importance["importance"], color="#8fe3bd"); ax.set_xlabel("Relative importance"); pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11.69, 8.27)); fig.text(.08, .88, "Prediction trace", fontsize=20, weight="bold")
        if prediction:
            text = f"Selected row: {prediction['row_index']}\nActual: {prediction['actual']:.2f} Wh\nPredicted: {prediction['predicted']:.2f} Wh\nAbsolute error: {prediction['absolute_error']:.2f} Wh\nConfidence: {prediction['confidence']:.1%}\n\nTop contributing features:\n" + "\n".join(f"• {k}: {v:+.3f}" for k, v in prediction["contributions"].items())
        else:
            text = "No prediction was run during this session."
        fig.text(.08, .76, text, fontsize=14, color="#253340", linespacing=1.65)
        pdf.savefig(fig); plt.close(fig)
    buf.seek(0)
    return buf.read()

# Main views
if page == "Overview":
    st.markdown("<div class='eyebrow'>ML workspace / live control room</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Make every<br><em>kilowatt</em> count.</div>", unsafe_allow_html=True)
    st.markdown("An explorable, explainable energy prediction system — connected to a real CSV dataset and a real scikit-learn training pipeline.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset rows", f"{len(get_active_df()):,}")
    c2.metric("Features", f"{len(build_features(get_active_df()).columns) - 1}")
    c3.metric("Production model", "Random Forest")
    c4.metric("Pipeline", "Connected")
    st.markdown("### Live process stream")
    render_terminal()
    st.markdown("### Lifecycle control room")
    steps = [("01", "Load source", "Read dataset/energy_dataset.csv or your uploaded CSV", "Dataset & Upload"), ("02", "Clean signal", "Remove duplicates, fill gaps, parse timestamps", "Cleaning"), ("03", "Prepare features", "Encode time context and split train/test", "Preprocessing"), ("04", "Train & compare", "Benchmark Linear Regression and Random Forest", "Model Lab"), ("05", "Explain output", "Generate prediction, error and feature contributions", "Live Prediction")]
    for n, title, desc, target_page in steps:
        with st.container(border=True):
            a, b, c = st.columns([.1, .3, 1])
            a.markdown(f"<div class='step-number'>{n}</div>", unsafe_allow_html=True); b.markdown(f"**{title}**"); c.markdown(desc)

elif page == "Dataset & Upload":
    st.markdown("<div class='eyebrow'>01 / data source</div><h1>One source of truth.</h1>", unsafe_allow_html=True)
    st.write(f"Default source: `{DATASET_PATH}` · [UCI download]({DATASET_URL})")
    uploaded = st.file_uploader("Upload a custom CSV to replace the active dataset", type=["csv"])
    if uploaded is not None and st.button("Use uploaded CSV", type="primary"):
        st.session_state.df = pd.read_csv(uploaded, sep=None, engine="python")
        st.session_state.results = None; st.session_state.prediction = None
        add_log(f"upload :: accepted {uploaded.name} ({len(st.session_state.df):,} rows)")
        st.rerun()
    df = get_active_df()
    c1, c2, c3, c4 = st.columns(4); c1.metric("Rows", f"{len(df):,}"); c2.metric("Columns", len(df.columns)); c3.metric("Missing cells", int(df.isna().sum().sum())); c4.metric("Duplicates", int(df.duplicated().sum()))
    st.markdown("### Dataset preview"); st.dataframe(df.head(25), use_container_width=True, height=370)
    with st.expander("Schema and summary statistics"):
        st.dataframe(pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str), "missing": df.isna().sum().values}), use_container_width=True)
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

elif page == "EDA Studio":
    df = get_active_df(); st.markdown("<div class='eyebrow'>02 / exploratory analysis</div><h1>See the building rhythm.</h1>", unsafe_allow_html=True)
    numeric = df.select_dtypes("number"); feature = st.selectbox("Choose a feature", [c for c in numeric.columns if c != TARGET] or [TARGET])
    left, right = st.columns(2)
    with left: st.plotly_chart(px.histogram(df, x=TARGET, nbins=40, title="Energy consumption distribution", template="plotly_dark"), use_container_width=True)
    with right: st.plotly_chart(px.scatter(df.sample(min(len(df), 3000)), x=feature, y=TARGET, trendline="ols", title=f"{feature} vs energy consumption", template="plotly_dark"), use_container_width=True)
    st.plotly_chart(px.imshow(numeric.corr(), color_continuous_scale="Tealgrn", title="Feature correlation heatmap", aspect="auto"), use_container_width=True)

elif page == "Cleaning":
    df = get_active_df(); cleaned = clean_dataframe(df); st.markdown("<div class='eyebrow'>03 / cleaning</div><h1>Make the signal dependable.</h1>", unsafe_allow_html=True)
    before, after = st.columns(2); before.metric("Before · missing cells", int(df.isna().sum().sum())); before.metric("Before · duplicates", int(df.duplicated().sum())); after.metric("After · missing cells", int(cleaned.isna().sum().sum())); after.metric("After · duplicates", int(cleaned.duplicated().sum()))
    st.dataframe(cleaned.head(25), use_container_width=True)
    st.info("Cleaning is deterministic: duplicate rows are removed, numeric gaps use median imputation, and date columns are parsed without fabricating records.")

elif page == "Preprocessing":
    st.markdown("<div class='eyebrow'>04 / preprocessing</div><h1>The path from raw to ready.</h1>", unsafe_allow_html=True)
    for n, title, detail in [("01", "Load source", "Read the active CSV into pandas"), ("02", "Clean values", "Median-impute numeric gaps and remove duplicates"), ("03", "Engineer time", "Extract hour/day context from date"), ("04", "Split", "80/20 train-test split with fixed seed 42"), ("05", "Fit", "Train two regressors on the same features")]:
        st.markdown(f"<div class='step'><div class='step-number'>{n}</div><div><b>{title}</b><div class='small-muted'>{detail}</div></div></div>", unsafe_allow_html=True)
    st.markdown("### Terminal trace"); render_terminal()

elif page == "Model Lab":
    df = get_active_df(); st.markdown("<div class='eyebrow'>05 / model lab</div><h1>Train the model you can defend.</h1>", unsafe_allow_html=True)
    st.write("Only two regressors are trained, as specified: Linear Regression baseline and Random Forest production candidate.")
    if st.button("Train / retrain on active dataset", type="primary"):
        with st.status("Running real training pipeline", expanded=True) as status:
            add_log("clean :: removing duplicates and filling numeric gaps"); st.write("Cleaning dataset…"); time.sleep(.25)
            add_log("features :: extracting hour and preparing matrices"); st.write("Building feature matrix…"); time.sleep(.25)
            add_log("train :: fitting Linear Regression and Random Forest"); st.write("Fitting models…")
            result = train_models(df, MODEL_PATH); st.session_state.results = result; st.session_state.prediction = None
            add_log(f"complete :: selected {result['best_model']} with R²={result['metrics'][result['best_model']]['r2']:.3f}")
            status.update(label="Training completed", state="complete")
    if st.session_state.results:
        result = st.session_state.results; metrics = pd.DataFrame(result["metrics"]).T.reset_index().rename(columns={"index": "model"}); st.dataframe(metrics, use_container_width=True)
        st.bar_chart(metrics.set_index("model")[["mae", "rmse"]])
        st.success(f"Selected production model: {result['best_model']}. It is persisted at `{MODEL_PATH}`.")
    else: st.warning("No active training run yet. Click the button to train against the current dataset.")

elif page == "Live Prediction":
    df = get_active_df(); st.markdown("<div class='eyebrow'>06 / live prediction</div><h1>Show the reasoning, not just the number.</h1>", unsafe_allow_html=True)
    if not st.session_state.results and MODEL_PATH.exists(): st.session_state.results = train_models(df, MODEL_PATH)
    if not st.session_state.results: st.warning("Train a model first in Model Lab."); st.stop()
    result = st.session_state.results; row_index = st.selectbox("Select a real row from the active dataset", list(range(min(len(df), 1000))))
    row = df.iloc[row_index]; st.dataframe(pd.DataFrame([row]), use_container_width=True)
    if st.button("Run live prediction", type="primary"):
        st.session_state.prediction = predict_row(row, result, row_index); add_log(f"predict :: row {row_index} scored {st.session_state.prediction['predicted']:.2f} Wh")
    if st.session_state.prediction:
        pred = st.session_state.prediction; a, b, c, d = st.columns(4); a.metric("Actual", f"{pred['actual']:.2f} Wh"); b.metric("Predicted", f"{pred['predicted']:.2f} Wh"); c.metric("Absolute error", f"{pred['absolute_error']:.2f} Wh"); d.metric("Confidence", f"{pred['confidence']:.1%}")
        st.markdown("### How this row was predicted")
        st.info("The model receives the selected row after the exact same cleaning and feature preparation used during training. The contribution list shows signed influence relative to the trained Random Forest's feature baseline; positive values push the estimate up and negative values push it down.")
        st.dataframe(pd.DataFrame({"feature": list(pred["contributions"].keys()), "signed contribution": list(pred["contributions"].values())}), use_container_width=True)

elif page == "Feature Importance":
    df = get_active_df(); st.markdown("<div class='eyebrow'>07 / explainability</div><h1>Know what moves the needle.</h1>", unsafe_allow_html=True)
    if not st.session_state.results: st.session_state.results = train_models(df, MODEL_PATH)
    importance = st.session_state.results["feature_importance"]; st.bar_chart(importance.set_index("feature").head(15)); st.dataframe(importance, use_container_width=True)

elif page == "Report":
    df = get_active_df(); st.markdown("<div class='eyebrow'>08 / report</div><h1>Package the proof.</h1>", unsafe_allow_html=True)
    if not st.session_state.results: st.session_state.results = train_models(df, MODEL_PATH)
    pdf = make_report(df, st.session_state.results, st.session_state.prediction)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "ecomind_ai_report.pdf").write_bytes(pdf)
    st.success("PDF report generated with dataset summary, EDA charts, feature importance and prediction trace.")
    st.download_button("Download PDF report", data=pdf, file_name="ecomind_ai_report.pdf", mime="application/pdf", type="primary")
    st.markdown("### Current terminal trace"); render_terminal()
