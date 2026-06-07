import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ---------------- LOAD MODELS ----------------
rf_model = joblib.load("rf_model.pkl")
rf_cls = joblib.load("rf_classifier.pkl")

encoders = joblib.load("encoders.pkl")

y_scaler = joblib.load("y_scaler.pkl")
X_scaler = joblib.load("X_scaler_revenue.pkl")
cls_scaler = joblib.load("cls_scaler.pkl")
mlb = joblib.load("mlb.pkl")

reg_features = X_scaler.feature_names_in_
cls_features = cls_scaler.feature_names_in_

# ---------------- UI ----------------
st.set_page_config(page_title="Marketing Predictor", layout="wide")
st.title("📊 Marketing Campaign Predictor")


# ------------SIDEBAR NAVIGATION PAGE BUTTONS ----------------


st.sidebar.title("📌 Navigation")

page = st.sidebar.selectbox(
    "Go to",
    ["Regression", "Classification"]
)
# ---------------- RESET BUTTON ----------------
REG_DEFAULTS = {
    "reg_brand": None,
    "reg_campaign_type": None,
    "reg_target_audience": None,
    "reg_language":None,
    "reg_customer_segment":None,
    "reg_channels": [],          # multiselect → empty list
    "reg_duration": 0,
    "reg_impressions": 0,
    "reg_clicks": 0,
    "reg_leads": 0,
    "reg_conversions": 0,
    "reg_acquisition_cost": 0.0,
    "reg_engagement_score": 0.0,
    "reg_calculated_roi": 0.0,
}
CLS_DEFAULTS = {
    "cls_brand": None,
    "cls_campaign_type": None,
    "cls_target_audience": None,
    "cls_language": None,
    "cls_customer_segment": None,
    "cls_channels": [],
    "cls_duration": 0,
    "cls_impressions": 0,
    "cls_clicks": 0,
    "cls_leads": 0,
    "cls_conversions": 0,
    "cls_acquisition_cost": 0.0,
    "cls_engagement_score": 0.0,
    "cls_revenue": 0.0
}

# ---------------- RESET BUTTON ----------------
def reset_all():
    for key, default in REG_DEFAULTS.items():
        st.session_state[key] = default   # overwrite with blank defaults
    # Add CLS defaults here too if needed
    for key, default in CLS_DEFAULTS.items():
        st.session_state[key] = default

st.sidebar.button("🔄 Reset", on_click=reset_all)



# PAGE 1 - REGRESSION

if page == "Regression":
    
    
    # ---------------- INPUTS ----------------
    brand = st.selectbox("Brand", encoders['Brand'].classes_, key="reg_brand")
    campaign_type = st.selectbox("Campaign Type", encoders['Campaign_Type'].classes_, key="reg_campaign_type")
    target_audience = st.selectbox("Target Audience", encoders['Target_Audience'].classes_, key="reg_target_audience")
    language = st.selectbox("Language", encoders['Language'].classes_, key="reg_language")
    customer_segment = st.selectbox("Customer Segment", encoders['Customer_Segment'].classes_, key="reg_customer_segment")

    channels = ['Email','Facebook','Google','Instagram','WhatsApp','YouTube']
    channel_used = st.multiselect("Select Channel Used",channels,key="reg_channels")

    duration = st.number_input("Duration", 0, key="reg_duration")
    impressions = st.number_input("Impressions", 0, key="reg_impressions")
    clicks = st.number_input("Clicks", 0, key="reg_clicks")
    leads = st.number_input("Leads", 0, key="reg_leads")
    conversions = st.number_input("Conversions", 0, key="reg_conversions")
    acquisition_cost = st.number_input("Acquisition Cost", 0.0, key="reg_acquisition_cost")
    engagement_score = st.number_input("Engagement Score", 0.0, key="reg_engagement_score")
    calculated_roi = st.number_input("Calculated ROI", value=0.0, key="reg_calculated_roi")

    today = datetime.today()
    # CHANNEL ENCODING 
    channel_array = mlb.transform([channel_used])[0]
    channel_dict = {
            f"Channel_Used_{col}": val
            for col, val in zip(mlb.classes_, channel_array)
        }

    
    
    if st.button("Predict Revenue 🚀"):
        if (
            brand is None or
            campaign_type is None or
            target_audience is None or
            language is None or
            customer_segment is None or
            len(channel_used) == 0
    ):
            st.warning("⚠️ Please complete all required inputs before making a prediction.")
            st.stop()

    
    
        # ---------------- INPUT FRAME ----------------
        input_data = pd.DataFrame([{
            "Brand": encoders['Brand'].transform([brand])[0],
            "Campaign_Type": encoders['Campaign_Type'].transform([campaign_type])[0],
            "Target_Audience": encoders['Target_Audience'].transform([target_audience])[0],
            "Language": encoders['Language'].transform([language])[0],
            "Customer_Segment": encoders['Customer_Segment'].transform([customer_segment])[0],

            "Duration": duration,
            "Impressions": impressions,
            "Clicks": clicks,
            "Leads": leads,
            "Conversions": conversions,
            "Acquisition_Cost": acquisition_cost,
            "Engagement_Score": engagement_score,
            'Calculated_ROI': calculated_roi,
            "Year": today.year,
            "Month": today.month,
            "Day": today.day,
            "Weekday": today.weekday(),
            **channel_dict
            
        }])
    
        X_reg = input_data.reindex(columns=reg_features, fill_value=0)
        X_reg_scaled = X_scaler.transform(X_reg)

        revenue_scaled = rf_model.predict(X_reg_scaled)
        revenue = y_scaler.inverse_transform(revenue_scaled.reshape(-1,1))[0][0]
        st.session_state["predicted_revenue"] = revenue

        roi = (revenue - acquisition_cost) / acquisition_cost if acquisition_cost > 0 else 0
        profit = revenue - acquisition_cost
        st.subheader("📊 Revenue + ROI + Profit Prediction")
        st.metric("💰 Revenue", f"₹ {revenue:,.2f}")
        st.metric("📊 ROI", f"{roi:.2f}")
        st.metric("📈 Profit", f"₹ {profit:,.2f}")

        if profit >= 0:
            st.success("✔ Profit Campaign")
        else:
            st.error("❌ Loss Campaign")

# PAGE 2 - CLASSIFICATION


elif page == "Classification":
        # ---------------- INPUTS ----------------
    brand = st.selectbox("Brand", encoders['Brand'].classes_, key="cls_brand")
    campaign_type = st.selectbox("Campaign Type", encoders['Campaign_Type'].classes_, key="cls_campaign_type")
    target_audience = st.selectbox("Target Audience", encoders['Target_Audience'].classes_, key="cls_target_audience")
    language = st.selectbox("Language", encoders['Language'].classes_, key="cls_language")
    customer_segment = st.selectbox("Customer Segment", encoders['Customer_Segment'].classes_, key="cls_customer_segment")

    channels = ['Email','Facebook','Google','Instagram','WhatsApp','YouTube']
    channel_used = st.multiselect("Select Channel Used",channels,key="cls_channels")

    duration = st.number_input("Duration", 0, key="cls_duration")
    impressions = st.number_input("Impressions", 0, key="cls_impressions")
    clicks = st.number_input("Clicks", 0, key="cls_clicks")
    leads = st.number_input("Leads", 0, key="cls_leads")
    conversions = st.number_input("Conversions", 0, key="cls_conversions")
    acquisition_cost = st.number_input("Acquisition Cost", 0.0, key="cls_acquisition_cost")
    engagement_score = st.number_input("Engagement Score", 0.0, key="cls_engagement_score")
    

    today = datetime.today()
    #  CHANNEL ENCODING 
    channel_array = mlb.transform([channel_used])[0]
    channel_dict = {
            f"Channel_Used_{col}": val
            for col, val in zip(mlb.classes_, channel_array)
        }
    revenue_input = st.number_input("Revenue", min_value=0.0,key="cls_revenue")


    if st.button("Predict Status 🚀"):
        if (
            brand is None or
            campaign_type is None or
            target_audience is None or
            language is None or
            customer_segment is None or
            len(channel_used) == 0
        ):
            st.warning("⚠️ Please complete all required inputs before making a prediction.")
            st.stop()


        cls_input = pd.DataFrame([{
            "Brand": encoders['Brand'].transform([brand])[0],
            "Campaign_Type": encoders['Campaign_Type'].transform([campaign_type])[0],
            "Target_Audience": encoders['Target_Audience'].transform([target_audience])[0],
            "Duration": duration,
            "Impressions": impressions,
            "Clicks": clicks,
            "Leads": leads,
            "Conversions": conversions,
            "Revenue": revenue_input,
            "Acquisition_Cost": acquisition_cost,
            "Language": encoders['Language'].transform([language])[0],
            "Engagement_Score": engagement_score,
            "Customer_Segment": encoders['Customer_Segment'].transform([customer_segment])[0],
            "Year": today.year,
            "Month": today.month,
            "Day": today.day,
            "Weekday": today.weekday(),
            **channel_dict
        }])
        X_cls = cls_input.reindex(columns=cls_features, fill_value=0)

        X_cls_scaled = cls_scaler.transform(X_cls)

        status_pred = rf_cls.predict(X_cls_scaled)[0]
        st.subheader("📈/📉 Status Prediction")

        if status_pred == 1:
            st.success("📈 Profit Campaign")
        else:
            st.error("📉 Loss Campaign")
        