import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0F0F11;
}

.main-header {
    text-align: center;
    padding: 3rem 0 1rem 0;
}

.main-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3.5rem;
    color: #F0EDE6;
    margin: 0;
    line-height: 1.1;
}

.main-header h1 em {
    font-style: italic;
    color: #C4A05A;
}

.main-header p {
    color: #6B6760;
    font-size: 1.1rem;
    font-weight: 300;
    margin-top: 0.75rem;
}

.divider {
    width: 60px;
    height: 2px;
    background: #C4A05A;
    margin: 1.5rem auto;
}

.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #C4A05A;
    margin-bottom: 1.2rem;
    margin-top: 2rem;
}

.card {
    background: #18181C;
    border: 1px solid #2A2A30;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

.result-box {
    background: #18181C;
    border: 1px solid #3A3020;
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-box .label {
    color: #6B6760;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.result-box .price {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #C4A05A;
    line-height: 1;
}

.result-box .note {
    color: #6B6760;
    font-size: 0.8rem;
    margin-top: 1rem;
}

.confidence-bar-bg {
    background: #2A2A30;
    border-radius: 99px;
    height: 6px;
    margin-top: 1.5rem;
}

.confidence-bar-fill {
    background: #C4A05A;
    border-radius: 99px;
    height: 6px;
    width: 82%;
}

.stat-row {
    display: flex;
    justify-content: space-around;
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid #2A2A30;
}

.stat-item .stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #F0EDE6;
}

.stat-item .stat-lbl {
    font-size: 0.7rem;
    color: #6B6760;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Streamlit widget overrides */
div[data-testid="stSlider"] > div > div > div > div {
    background: #C4A05A !important;
}

div[data-testid="stSlider"] > div > div > div {
    background: #2A2A30 !important;
}

div[data-testid="stSlider"] p {
    color: #9B9790 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #C4A05A, #A07C38) !important;
    color: #0F0F11 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    margin-top: 1rem !important;
    box-shadow: 0 4px 20px rgba(196,160,90,0.25) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(196,160,90,0.35) !important;
}

div[data-testid="stNumberInput"] input {
    background: #1E1E24 !important;
    border: 1px solid #2A2A30 !important;
    border-radius: 10px !important;
    color: #F0EDE6 !important;
    font-family: 'DM Sans', sans-serif !important;
}

div[data-testid="stSelectbox"] > div > div {
    background: #1E1E24 !important;
    border: 1px solid #2A2A30 !important;
    border-radius: 10px !important;
    color: #F0EDE6 !important;
}

label {
    font-family: 'DM Sans', sans-serif !important;
    color: #9B9790 !important;
    font-size: 0.9rem !important;
}

.stAlert {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "model", "house_price_model.pkl")
    return joblib.load(model_path)

model = load_model()

st.markdown("""
<div class="main-header">
    <h1>House Price <em>Predictor</em></h1>
    <div class="divider"></div>
    <p>Enter your property details and get an instant AI-powered estimate</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Property basics</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        OverallQual = st.slider("Overall Quality", 1, 10, 7,
            help="Rate the overall material and finish quality (1=Poor, 10=Excellent)")
        YearBuilt = st.slider("Year Built", 1870, 2024, 1995)
        GrLivArea = st.number_input("Living Area (sq ft)", 500, 6000, 1500, step=50)
        TotalBsmtSF = st.number_input("Basement Area (sq ft)", 0, 3000, 850, step=50)

    with col2:
        OverallCond = st.slider("Overall Condition", 1, 10, 6,
            help="Rate the overall condition (1=Poor, 10=Excellent)")
        YearRemodAdd = st.slider("Year Remodeled", 1950, 2024, 2005)
        GarageArea = st.number_input("Garage Area (sq ft)", 0, 1500, 480, step=50)
        LotArea = st.number_input("Lot Area (sq ft)", 1000, 50000, 8500, step=100)

    st.markdown('<div class="section-label">Rooms & location</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        TotRmsAbvGrd = st.slider("Total Rooms Above Ground", 2, 14, 7)
        Neighborhood = st.selectbox("Neighborhood", [
            'NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst',
            'NridgHt', 'Gilbert', 'Sawyer', 'NWAmes', 'SawyerW',
            'BrkSide', 'Crawfor', 'Mitchel', 'NoRidge', 'Timber',
            'IDOTRR', 'ClearCr', 'StoneBr', 'SWISU', 'Blmngtn',
            'MeadowV', 'BrDale', 'Veenker', 'NPkVill', 'Blueste'
        ])

    with col4:
        HouseStyle = st.selectbox("House Style", [
            '1Story', '2Story', '1.5Fin', 'SLvl',
            'SFoyer', '1.5Unf', '2.5Unf', '2.5Fin'
        ])
        SaleCondition = st.selectbox("Sale Condition", [
            'Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family'
        ])

    st.markdown('</div>', unsafe_allow_html=True)
    predict_btn = st.button("Estimate Price →")

with right:
    if predict_btn:
        input_data = pd.DataFrame([{
            'LotArea'      : LotArea,
            'OverallQual'  : OverallQual,
            'OverallCond'  : OverallCond,
            'TotalBsmtSF'  : TotalBsmtSF,
            'TotRmsAbvGrd' : TotRmsAbvGrd,
            'GarageArea'   : GarageArea,
            'GrLivArea'    : GrLivArea,
            'YearBuilt'    : YearBuilt,
            'YearRemodAdd' : YearRemodAdd,
            'Neighborhood' : Neighborhood,
            'HouseStyle'   : HouseStyle,
            'SaleCondition': SaleCondition
        }])

        prediction = np.expm1(model.predict(input_data))[0]
        low = prediction * 0.92
        high = prediction * 1.08
        age = 2024 - YearBuilt

        st.markdown(f"""
        <div class="result-box">
            <div class="label">Estimated Market Value</div>
            <div class="price">${prediction:,.0f}</div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill"></div>
            </div>
            <div class="note">Confidence range: ${low:,.0f} – ${high:,.0f}</div>
            <div class="stat-row">
                <div class="stat-item">
                    <div class="stat-val">{age}y</div>
                    <div class="stat-lbl">Age</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val">${prediction/GrLivArea:,.0f}</div>
                    <div class="stat-lbl">Per sq ft</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val">{TotRmsAbvGrd}</div>
                    <div class="stat-lbl">Rooms</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:2rem">Property summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        summary_data = {
            "Detail": ["Style", "Neighborhood", "Sale Type", "Quality Score", "Condition Score"],
            "Value": [HouseStyle, Neighborhood, SaleCondition, f"{OverallQual}/10", f"{OverallCond}/10"]
        }
        st.dataframe(
            pd.DataFrame(summary_data),
            hide_index=True,
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="height: 100%; display: flex; flex-direction: column; align-items: center;
                    justify-content: center; text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🏡</div>
            <div style="font-family: 'DM Serif Display', serif; font-size: 1.5rem;
                        color: #F0EDE6; margin-bottom: 0.5rem;">Ready to estimate</div>
            <div style="font-size: 0.9rem; font-weight: 300; max-width: 260px;
                        line-height: 1.6; color: #6B6760;">
                Fill in the property details on the left and click
                <strong style="color:#C4A05A">Estimate Price</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#3A3A42; font-size:0.75rem;
            padding: 2rem 0; letter-spacing: 0.05em;">
    Powered by XGBoost · Trained on Ames Housing Dataset
</div>
""", unsafe_allow_html=True)