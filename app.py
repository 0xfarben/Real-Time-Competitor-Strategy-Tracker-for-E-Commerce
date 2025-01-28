import os
import json
import requests
import pandas as pd
import streamlit as st 
import plotly.express as px
from datetime import datetime
from transformers import pipeline
from statsmodels.tsa.arima.model import ARIMA

# Constants
GROQ_API_KEY = "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"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
SLACK_WEBHOOK = "https://hooks.slack.com/services/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"

# Helper Functions
def truncate_text(text, max_length=512):
    return text[:max_length]

def load_csv(file_path):
    """Load a CSV file with robust error handling."""
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return pd.DataFrame()
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()
    
# NOTE - Use this link for manual testing of Sentiments -> 
# https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english

def clean_competitor_data(data):
    """Clean and preprocess competitor data."""
    # data["Discount"] = data["Discount"].astype(float)
    data = data.dropna(subset=["Price", "MRP", "Discount"])
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data.dropna(subset=["Date"], inplace=True)
    data.set_index("Date", inplace=True)
    return data


def analyze_sentiment(reviews):
    """Analyze sentiment of reviews using a pre-trained model."""
    try:
        sentiment_pipeline = pipeline(
            task="text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
        res = sentiment_pipeline(reviews)
        return res
    except Exception as e:
        st.error(f"Sentiment analysis failed: {e}")
        return None


def forecast_discounts_arima(data, future_days=5):
    """
    Forecast future discounts using the ARIMA model.
    Args:
        data (pd.DataFrame): A DataFrame containing a "Discount" column and a time-based index.
        future_days (int): Number of future days to forecast.
    Returns:
        pd.DataFrame: A DataFrame containing forecasted discounts with future dates as the index.
    """
    if len(data) < 6:  # At least 6 rows for ARIMA to work with order=(5, 1, 0)
        st.warning("⚠️ Insufficient data for ARIMA forecasting. At least 6 rows are required.")
        return None

    try:
        # Ensure the index is a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            st.error("Index must be a DatetimeIndex for ARIMA forecasting.")
            return None

        # Fit the ARIMA model
        discount_series = data["Discount"]
        model = ARIMA(discount_series, order=(5, 1, 0))
        model_fit = model.fit()

        # Generate forecast
        forecast = model_fit.forecast(steps=future_days)

        # Create a DataFrame for the forecasted values
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast})
        forecast_df.set_index("Date", inplace=True)

        return forecast_df
    except Exception as e:
        st.error(f"ARIMA forecasting failed: {e}")
        return None

def send_to_slack(message):
    """Send a message to Slack via a webhook."""
    payload = {"text": message}
    try:
        response = requests.post(
            SLACK_WEBHOOK,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        if response.status_code != 200:
            st.error(f"Slack API error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Failed to send message to Slack: {e}")

def generate_strategy_recommendation(product_name, product_data, sentiment):
    """
    Generate strategic recommendations using OpenAI GPT-4, Groq AI, and Gemini AI.
    
    Args:
        product_name (str): Name of the product.
        competitor_data (str): Competitor pricing and discount data.
        sentiment (str): Sentiment analysis data.
        api_keys (dict): Dictionary containing API keys for OpenAI and Groq AI.
    
    Returns:
        dict: A dictionary containing responses from OpenAI GPT-4, Groq AI, and Gemini AI.
    """
    if competitor_data is None or sentiment is None:
        return {"error": "Insufficient data to generate recommendations."}

    date = datetime.now()
    # Mapping sentiment labels if not already mapped
    sentiment_mapping = {
        'LABEL_0': 'Negative',
        'LABEL_1': 'Neutral',
        'LABEL_2': 'Positive'
    }

    # Check if sentiment_mapped is defined or set a default value
    if 'sentiment_mapped' not in locals():
        sentiment_mapped = []  # Initialize as an empty list or appropriate default

    # If you have multiple results
    if isinstance(sentiment_mapped, list) and sentiment_mapped:
        sentiment_mapped = [
            {
                'label': sentiment_mapping.get(item['label'], 'Unknown'),
                'score': item['score']
            } for item in sentiment_mapped
        ]
    elif isinstance(sentiment_mapped, str):  # For a single result
        sentiment_mapped = sentiment_mapping.get(sentiment_mapped, 'Unknown')
    else:
        sentiment_mapped = 'Unknown'

    # Format sentiment data for the prompt
    if isinstance(sentiment_mapped, list):
        sentiment_details = [
            f"{item['label']} (Score: {item['score']:.2f})"
            for item in sentiment_mapped
        ]
        sentiment_summary = ", ".join(sentiment_details)
    else:
        sentiment_summary = sentiment_mapped  # For single result or 'Unknown'

    print(product_data)
    # Final Prompt
    prompt = f"""
        You are a highly skilled business strategist specializing in **Indian e-commerce**. Based on the following details, suggest strategic recommendations:

        ### **Product Details**
        - **Product Name**: {product_name}
        - **Competitor Data** (Prices are in INR ₹, NOT in USD $): {product_data}
        - **Sentiment Analysis** (Mapped labels: {sentiment_summary})
        - **Today's Date**: {str(date)}

        ### **Task:**
        1. **Pricing Strategy:**  
        - Analyze competitor pricing trends and suggest optimal pricing (considering INR ₹ as the currency).  
        - Ensure recommendations **do not assume USD**; prices should be in INR ₹.  
        - Suggest a discounting approach based on **predicted discounts** and **customer sentiment trends**.

        2. **Promotional Campaign Ideas:**  
        - Recommend marketing and promotional tactics based on sentiment trends (e.g., more discounts if sentiment is negative).  
        - Identify product positioning strategies to **differentiate from competitors**.

        3. **Customer Satisfaction Recommendations:**  
        - Use sentiment insights to **improve product perception**.  
        - Provide suggestions based on the **most common negative feedback themes**.

        ### **Important Considerations:**
        - Do **not assume USD ($) for pricing**. All monetary values are in **Indian Rupees (₹)**.
        - Use sentiment labels correctly (Negative, Neutral, Positive) instead of generic AI labels (LABEL_0, LABEL_1, LABEL_2).
        - Provide **actionable**, **data-driven** recommendations specific to the Indian e-commerce market.
    """
    try:
        groq_data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "llama3-8b-8192",
            "temperature": 0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        groq_response = requests.post(
            GROQ_ENDPOINT,
            json=groq_data,
            headers=headers,
        )
        groq_response.raise_for_status()  # Ensure no HTTP errors
        responses = groq_response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        responses = f"Error generating recommendations with Groq AI: {e}"


    return responses


############# STARTING HERE #################
# Load Data
competitor_data = load_csv(r"Products.csv")
reviews_data = load_csv(r"Reviews.csv")
print(competitor_data.columns)
print(competitor_data.head())
if not competitor_data.empty:
    competitor_data = clean_competitor_data(competitor_data)

if not reviews_data.empty:
    reviews_data["Review_Date"] = pd.to_datetime(reviews_data["Review_Date"], errors="coerce")
    reviews_data.dropna(subset=["Review_Date", "Rating", "Review_Text"], inplace=True)

# Streamlit App Setup
st.set_page_config(
    page_title="E-Commerce Competitor Strategy Dashboard",
    page_icon="https://iili.io/2QnnboB.md.jpg",
    layout="wide"
)

# **Page Title (Aligned Left)**
st.markdown("<h1 style='text-align: center;'> E-Commerce 🛒 Competitor Strategy Dashboard</h1>", unsafe_allow_html=True)

# **Product Selection Panel (Full Width)**
st.markdown("<h3 style='text-align: center;'>📌 Select a Product to Analyze 📌 </h3>", unsafe_allow_html=True)

# Create a full-width row layout for dropdown
col1, col2, col3 = st.columns([1, 4, 1])  # Middle column wider
with col2:
    products = reviews_data["Product_Name"].unique().tolist() if not reviews_data.empty else []
    selected_product = st.selectbox("", products, key="product_select") if products else None  # Removed the default label


# Display Product Analysis Section
if selected_product:
    product_reviews = reviews_data[reviews_data["Product_Name"] == selected_product]
    product_data = competitor_data[competitor_data["Product_Name"] == selected_product]

    st.divider()
    st.header(f"📈 COMPETITOR ANALYSIS")
    # st.header(")
    st.subheader(f"📊 Product: {selected_product}")

    # Sentiment Analysis
    if not product_reviews.empty:
        asin = product_data["Product_ASIN"].values[0]  # Get ASIN from CSV
        
        if asin:
            # **Generate Amazon Image URL**
            product_image_url = f"https://images.amazon.com/images/P/{asin}.jpg"
            # **Display Product Image with medium size (adjust width as needed)**
            st.markdown(f"<div style='text-align: center;'><img src='{product_image_url}' alt='{selected_product}' width='400'/></div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ ASIN not found for this product. Image unavailable.")
        st.divider()

        # **Display Product Image**
        st.subheader("🗣️ CUSTOMER SENTIMENT ANALYSIS")
        
        # Extract reviews and truncate text if necessary
        reviews = product_reviews["Review_Text"].apply(lambda x: truncate_text(x)).tolist()

        # Analyze sentiment with a loading spinner
        with st.spinner("Analyzing sentiment..."):
            sentiments = analyze_sentiment(reviews)
        
        if sentiments:
            sentiment_df = pd.DataFrame(sentiments)
            label_mapping = {
                "LABEL_0": "Negative",
                "LABEL_1": "Neutral",
                "LABEL_2": "Positive"
            }
            sentiment_df["label"] = sentiment_df["label"].map(label_mapping)

            # Define custom colors for each sentiment
            color_map = {
                "Negative": "#f54248",  # Red for negative sentiment
                "Neutral": "#42ddf5",   # Blue for neutral sentiment
                "Positive": "#69f542"   # Green for positive sentiment
            }

            # Create the bar chart with custom colors
            fig = px.bar(
                sentiment_df,
                x="label",
                color="label",
                # title="Sentiment Analysis",
                color_discrete_map=color_map
            )
            st.plotly_chart(fig)

            # Display the top N reviews with a numbered index starting from 1
            num_reviews_to_display = 10
            reviews_to_display = product_reviews[["Review_Title", "Review_Text"]].head(num_reviews_to_display)
            reviews_to_display.index = range(1, len(reviews_to_display) + 1)

            # Create a full-width container for the reviews
            with st.container():
                st.dataframe(reviews_to_display, use_container_width=True)
   

    st.divider()


    # Competitor Data Analysis
    # Check if there's sufficient data for forecasting
    if len(product_data) < 6:
        st.warning("⚠️ Insufficient data for ARIMA forecasting. At least 6 data points are required.")
    else:
        # Perform forecasting with a loading spinner
        with st.spinner("Forecasting discounts using ARIMA..."):
            forecast_df = forecast_discounts_arima(product_data, future_days=5)
        
        if forecast_df is not None:
            st.subheader("❄️ FORECASTED DISCOUNTS")
            st.line_chart(forecast_df["Predicted_Discount"])
            # st.write(forecast_df)
            

            # Create a full-width container for the Discounts table.
            with st.container():
                st.dataframe(forecast_df, use_container_width=True)
        else:
            st.error("Forecasting failed. Please check your data.")

    st.divider()

    # Strategic Recommendations
    if not product_reviews.empty and not product_data.empty:
        st.subheader("📌 STRATEGIC RECOMMENDATIONS 🧠 🌌")
        with st.spinner("Generating strategic recommendations..."):
            
            # Generate strategy recommendation with the correct data format
            response = generate_strategy_recommendation(
                product_name=selected_product,
                product_data=product_data.reset_index(inplace=True),  # Replace competitor_data with historical_data
                sentiment=sentiments
            )
                    
        if response:
            st.success(response)
            
            # Send recommendations to Slack (optional)
            with st.spinner("Sending recommendations to Slack..."):
                slack_message = "\n\n".join(response)
                send_to_slack(slack_message)
        else:
            st.error("Failed to generate recommendations.")
else:
    st.warning("⚠️ Please select a product to analyze.")

st.markdown("<h6 style='text-align: center;'>© 2025 MarketSense: E-Commerce Strategy Dashboard</h6>", unsafe_allow_html=True)
