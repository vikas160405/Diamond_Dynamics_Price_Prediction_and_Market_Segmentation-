"""
💎 Diamond Dynamics: Price Prediction & Market Segmentation
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import os

# ─────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────
st.set_page_config(
    page_title="💎 Diamond Dynamics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# Custom CSS Styling
# ─────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0d1117; }

    /* Header styling */
    .diamond-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #e94560;
        box-shadow: 0 0 30px rgba(233, 69, 96, 0.3);
    }
    .diamond-header h1 { color: #e94560; font-size: 2.5rem; margin: 0; }
    .diamond-header p  { color: #a8b2d8; font-size: 1rem; margin: 0.5rem 0 0 0; }

    /* Card style */
    .card {
        background: #161b22;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #30363d;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #0f3460, #16213e);
        border: 2px solid #e94560;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 0 20px rgba(233,69,96,0.4);
    }
    .result-box .price { color: #00d4aa; font-size: 2.5rem; font-weight: bold; }
    .result-box .label { color: #a8b2d8; font-size: 1rem; }

    /* Cluster badge */
    .cluster-premium    { background:#e74c3c; color:white; padding:0.5rem 1rem;
                          border-radius:20px; font-weight:bold; font-size:1.1rem; }
    .cluster-midrange   { background:#3498db; color:white; padding:0.5rem 1rem;
                          border-radius:20px; font-weight:bold; font-size:1.1rem; }
    .cluster-affordable { background:#2ecc71; color:white; padding:0.5rem 1rem;
                          border-radius:20px; font-weight:bold; font-size:1.1rem; }

    /* Metric tiles */
    .metric-tile {
        background:#21262d; border-radius:10px; padding:1rem;
        text-align:center; border:1px solid #30363d;
    }
    .metric-tile .val { color:#58a6ff; font-size:1.6rem; font-weight:bold; }
    .metric-tile .lbl { color:#8b949e; font-size:0.85rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161b22 !important; }
    .stSelectbox label, .stSlider label, .stNumberInput label { color: #c9d1d9 !important; }
    .stButton>button {
        background: linear-gradient(135deg, #e94560, #c0392b);
        color: white; border: none; border-radius: 8px;
        font-size: 1.1rem; font-weight: bold;
        padding: 0.6rem 2rem; width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(233,69,96,0.4); }

    h2, h3 { color: #c9d1d9 !important; }
    p { color: #8b949e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Load Models & Metadata
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    files = {
        'regressor':        'best_regression_model.pkl',
        'scaler':           'scaler.pkl',
        'scaler_cluster':   'scaler_cluster.pkl',
        'kmeans':           'kmeans_model.pkl',
        'encoder':          'ordinal_encoder.pkl',
        'cluster_names':    'cluster_names.pkl',
    }
    for key, fname in files.items():
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                models[key] = pickle.load(f)
        else:
            models[key] = None
    if os.path.exists('model_meta.json'):
        with open('model_meta.json') as f:
            models['meta'] = json.load(f)
    else:
        models['meta'] = None
    return models

models = load_models()

# Default metadata (fallback if models not trained yet)
CUT_OPTIONS     = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
COLOR_OPTIONS   = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
CLARITY_OPTIONS = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']
USD_TO_INR      = 83.5

CUT_MAP     = {v: i for i, v in enumerate(CUT_OPTIONS)}
COLOR_MAP   = {v: i for i, v in enumerate(['J','I','H','G','F','E','D'])}
CLARITY_MAP = {v: i for i, v in enumerate(['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])}

CLUSTER_ICONS = {
    'Premium Heavy Diamonds':      ('💎', "#b93ce7", 'cluster-premium'),
    'Mid-range Balanced Diamonds': ('🔷', '#3498db', 'cluster-midrange'),
    'Affordable Small Diamonds':   ('💠', '#2ecc71', 'cluster-affordable'),
}
CLUSTER_DESCRIPTIONS = {
    'Premium Heavy Diamonds':      'Large, high-carat, premium-grade stones. Best cut quality and high brilliance.',
    'Mid-range Balanced Diamonds': 'Balanced between size and price. Ideal for engagement rings with good value.',
    'Affordable Small Diamonds':   'Small, budget-friendly stones. Great for everyday jewelry and gift items.',
}

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.markdown("""
<div class="diamond-header">
    <h1>💎 Diamond Dynamics</h1>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Sidebar – Input Form
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💍 Diamond Attributes")
    st.markdown("---")

    st.markdown("### ⚖️ Physical Properties")
    carat = st.slider("Carat Weight", min_value=0.2, max_value=5.0,
                      value=1.0, step=0.01)
    x     = st.number_input("Length - x (mm)", min_value=0.1, max_value=15.0,
                             value=6.0, step=0.1)
    y     = st.number_input("Width  - y (mm)", min_value=0.1, max_value=15.0,
                             value=6.0, step=0.1)
    z     = st.number_input("Depth  - z (mm)", min_value=0.1, max_value=10.0,
                             value=3.7, step=0.1)
    depth_val = st.slider("Depth %", min_value=40.0, max_value=80.0,
                          value=62.0, step=0.1)
    table_val = st.slider("Table %", min_value=43.0, max_value=100.0,
                          value=57.0, step=0.5)

    st.markdown("### 🎨 Quality Grades")
    cut     = st.selectbox("Cut Quality",     CUT_OPTIONS,     index=4)
    color   = st.selectbox("Color Grade",     COLOR_OPTIONS,   index=0)
    clarity = st.selectbox("Clarity Grade",   CLARITY_OPTIONS, index=3)

    st.markdown("---")
    st.markdown("**Diamond Grading Guide:**")
    st.markdown("- **Cut**: Fair → Ideal (best)")
    st.markdown("- **Color**: D (colorless) → J (light yellow)")
    st.markdown("- **Clarity**: IF (flawless) → I1 (included)")

# ─────────────────────────────────────────
# Feature Preparation Helper
# ─────────────────────────────────────────
def prepare_features(carat, cut, color, clarity, depth_val, table_val, x, y, z):
    """Prepare feature vector for regression model."""
    # Derived features
    volume    = x * y * z
    dim_ratio = (x + y) / (2 * z) if z > 0 else 0.0

    # Ordinal encoding
    cut_enc     = CUT_MAP.get(cut, 2)
    color_enc   = COLOR_MAP.get(color, 3)
    clarity_enc = CLARITY_MAP.get(clarity, 4)

    # features_final = ['carat','cut','color','clarity','depth','table','volume','dimension_ratio']
    features = np.array([[carat, cut_enc, color_enc, clarity_enc,
                          depth_val, table_val, volume, dim_ratio]])
    return features

def prepare_cluster_features(carat, cut, color, clarity, depth_val, table_val, x, y, z):
    """Prepare feature vector for clustering model."""
    volume    = x * y * z
    cut_enc   = CUT_MAP.get(cut, 2)
    color_enc = COLOR_MAP.get(color, 3)
    clarity_enc = CLARITY_MAP.get(clarity, 4)
    # cluster_features = ['carat','cut','color','clarity','depth','table','volume']
    features  = np.array([[carat, cut_enc, color_enc, clarity_enc,
                           depth_val, table_val, volume]])
    return features

# ─────────────────────────────────────────
# MAIN CONTENT — Two Tabs
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💰 Price Prediction", "🔵 Market Segmentation", "📊 About Dataset"])

# ═══════════════════════════════════════════
# TAB 1 — Price Prediction
# ═══════════════════════════════════════════
with tab1:
    col_info, col_predict = st.columns([1.3, 1])

    with col_info:
        st.markdown("### 📋 Selected Diamond Profile")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="val">{carat}</div>
                <div class="lbl">Carats</div>
            </div>""", unsafe_allow_html=True)
        with metrics_col2:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="val">{cut}</div>
                <div class="lbl">Cut</div>
            </div>""", unsafe_allow_html=True)
        with metrics_col3:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="val">{color}</div>
                <div class="lbl">Color</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="val">{clarity}</div>
                <div class="lbl">Clarity</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            volume_display = round(x * y * z, 2)
            st.markdown(f"""
            <div class="metric-tile">
                <div class="val">{volume_display}</div>
                <div class="lbl">Volume (mm³)</div>
            </div>""", unsafe_allow_html=True)
        with mc3:
            dim_r = round((x + y) / (2 * z), 2) if z > 0 else 0
            st.markdown(f"""
            <div class="metric-tile">
                <div class="val">{dim_r}</div>
                <div class="lbl">Dim Ratio</div>
            </div>""", unsafe_allow_html=True)

        # Carat category badge
        if carat < 0.5:   cat_lbl, cat_col = "🪨 Light",  "#8b949e"
        elif carat < 1.5: cat_lbl, cat_col = "💎 Medium", "#58a6ff"
        else:             cat_lbl, cat_col = "👑 Heavy",  "#e94560"
        st.markdown(f"<br><center><span style='background:{cat_col};color:white;padding:6px 16px;"
                    f"border-radius:20px;font-weight:bold;font-size:0.9rem'>{cat_lbl} Carat</span>"
                    f"</center>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_predict:
        st.markdown("### 💰 Predict Diamond Price")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if st.button("🔮 Predict Price", key="btn_price"):
            feat = prepare_features(carat, cut, color, clarity, depth_val, table_val, x, y, z)

            if models['regressor'] is not None:
                # Scale features
                if models['scaler'] is not None:
                    feat_scaled = models['scaler'].transform(feat)
                else:
                    feat_scaled = feat

                pred_inr = float(models['regressor'].predict(feat_scaled)[0])
                pred_usd = pred_inr / USD_TO_INR
            else:
                # Fallback formula (approx) if model not yet trained
                base = carat * 4000 * USD_TO_INR
                cut_mult = {'Fair':0.85,'Good':0.90,'Very Good':0.95,'Premium':1.0,'Ideal':1.05}
                pred_inr = base * cut_mult.get(cut, 1.0)
                pred_usd = pred_inr / USD_TO_INR
                st.warning("⚠️ Model not found — showing estimated price. Run the notebook first.")

            st.markdown(f"""
            <div class="result-box">
                <div class="label">Predicted Price (INR)</div>
                <div class="price">₹{pred_inr:,.0f}</div>
                <br>
                <div class="label">≈ USD ${pred_usd:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            # Price range insight
            st.markdown("<br>", unsafe_allow_html=True)
            if pred_inr < 50000:       tier, tip = "Budget",   "💡 Great value for everyday jewelry"
            elif pred_inr < 3_00_000:  tier, tip = "Mid-range","💡 Popular for engagement rings"
            elif pred_inr < 10_00_000: tier, tip = "Premium",  "💡 Investment-grade diamond"
            else:                       tier, tip = "Luxury",   "💡 Collector-grade diamond"
            st.info(f"**{tier} Tier** — {tip}")

        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 2 — Market Segmentation
# ═══════════════════════════════════════════
with tab2:
    col_seg, col_desc = st.columns([1, 1.2])

    with col_seg:
        st.markdown("### 🔵 Market Cluster Prediction")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("Click below to identify which market segment this diamond belongs to.")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔍 Identify Market Segment", key="btn_cluster"):
            feat = prepare_cluster_features(carat, cut, color, clarity,
                                            depth_val, table_val, x, y, z)

            if models['kmeans'] is not None and models['scaler_cluster'] is not None:
                feat_scaled = models['scaler_cluster'].transform(feat)
                cluster_id  = int(models['kmeans'].predict(feat_scaled)[0])
                cn_map = models['cluster_names'] if models['cluster_names'] else {}
                cluster_label = cn_map.get(cluster_id, cn_map.get(str(cluster_id),
                                           'Mid-range Balanced Diamonds'))
            else:
                # Heuristic fallback
                if carat > 1.5:   cluster_label = 'Premium Heavy Diamonds'
                elif carat < 0.5: cluster_label = 'Affordable Small Diamonds'
                else:             cluster_label = 'Mid-range Balanced Diamonds'
                st.warning("⚠️ KMeans model not found — using heuristic. Run the notebook first.")

            icon, color_hex, css_class = CLUSTER_ICONS.get(
                cluster_label, ('🔷', '#3498db', 'cluster-midrange'))

            st.markdown(f"""
            <div class="result-box">
                <div class="label">Market Segment</div>
                <br>
                <div style="font-size:3rem">{icon}</div>
                <br>
                <span class="{css_class}">{cluster_label}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            desc = CLUSTER_DESCRIPTIONS.get(cluster_label, '')
            st.success(f"📌 {desc}")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_desc:
        st.markdown("### 📖 Market Segment Reference")
        for name, (icon, color_hex, _) in CLUSTER_ICONS.items():
            st.markdown(f"""
            <div style="background:#161b22;border-left:4px solid {color_hex};
                        border-radius:8px;padding:1rem;margin-bottom:1rem;">
                <h4 style="color:{color_hex};margin:0">{icon} {name}</h4>
                <p style="color:#8b949e;margin:0.5rem 0 0 0">
                    {CLUSTER_DESCRIPTIONS[name]}
                </p>
            </div>
            """, unsafe_allow_html=True)

        
# ═══════════════════════════════════════════
# TAB 3 — About Dataset
# ═══════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Diamond Dataset Overview")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📋 Dataset Stats")
        stats_data = {
            "Attribute": ["Total Records", "Total Features", "Target Variable",
                          "Price Range", "Carat Range", "Domain"],
            "Value":     ["53,940 rows", "10 columns", "Price (USD → INR)",
                          "$326 – $18,823", "0.2 – 5.01", "Luxury Goods / E-Commerce"]
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🔑 Feature Descriptions")
        feat_data = {
            "Feature":     ["carat","cut","color","clarity","depth","table","x","y","z"],
            "Type":        ["Numeric","Ordinal","Ordinal","Ordinal",
                            "Numeric","Numeric","Numeric","Numeric","Numeric"],
            "Description": [
                "Weight in carats", "Cut quality (5 levels)",
                "Color grade D–J", "Clarity grade IF–I1",
                "Total depth %", "Table width %",
                "Length (mm)", "Width (mm)", "Height (mm)"
            ]
        }
        st.dataframe(pd.DataFrame(feat_data), hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    



