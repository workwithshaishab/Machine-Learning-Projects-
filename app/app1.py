import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("../model/house_price_model.pkl")

st.title("House Price Predictor")
st.write("Enter the house details below to get a predicted price.")

col1, col2 = st.columns(2)

with col1:
    OverallQual  = st.slider("Overall Quality", 1, 10, 5)
    GrLivArea    = st.number_input("Living Area (sq ft)", 500, 6000, 1500)
    TotalBsmtSF  = st.number_input("Basement Area (sq ft)", 0, 3000, 800)
    GarageArea   = st.number_input("Garage Area (sq ft)", 0, 1500, 400)
    YearBuilt    = st.slider("Year Built", 1870, 2025, 1990)

with col2:
    YearRemodAdd = st.slider("Year Remodeled", 1950, 2025, 2000)
    LotArea      = st.number_input("Lot Area (sq ft)", 1000, 50000, 8000)
    TotRmsAbvGrd = st.slider("Total Rooms", 2, 14, 6)
    OverallCond  = st.slider("Overall Condition", 1, 10, 5)
    Neighborhood = st.selectbox("Neighborhood", [
        'NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst',
        'NridgHt', 'Gilbert', 'Sawyer', 'NWAmes', 'SawyerW',
        'BrkSide', 'Crawfor', 'Mitchel', 'NoRidge', 'Timber',
        'IDOTRR', 'ClearCr', 'StoneBr', 'SWISU', 'Blmngtn',
        'MeadowV', 'BrDale', 'Veenker', 'NPkVill', 'Blueste'
    ])

HouseStyle    = st.selectbox("House Style", [
    '1Story', '2Story', '1.5Fin', 'SLvl', 'SFoyer', '1.5Unf', '2.5Unf', '2.5Fin'
])
SaleCondition = st.selectbox("Sale Condition", [
    'Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family'
])

if st.button("Predict Price"):
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
    st.success(f"Estimated House Price: ${prediction:,.0f}")