import importlib.util
import os
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA


############# CONSTANTS #############
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
MIN_FORECAST_OBSERVATIONS = 8


############# HELPER FUNCTIONS #############
def get_config_value(name, default=""):
    """Read configuration from the environment or Streamlit secrets."""
    environment_value = os.getenv(name)
    if environment_value:
        return environment_value.strip()

    try:
        secret_value = st.secrets.get(name, default)
    except Exception:
        # Streamlit raises when no secrets file exists. Environment variables
        # remain sufficient for local development.
        secret_value = default

    return str(secret_value).strip() if secret_value is not None else default


def load_csv(file_path):
    """Load a CSV file with robust error handling."""
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return pd.DataFrame()
        return pd.read_csv(file_path)
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Failed to load {file_path}: {error}")
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    """Load and cache the PyTorch sentiment model."""
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            "PyTorch is not installed. Run "
            "`python -m pip install -r requirements.txt` and restart the app."
        )

    # Import lazily so an incomplete environment produces a clear dashboard
    # message instead of warnings and a `torch is not defined` exception.
    from transformers import pipeline

    return pipeline(
        task="text-classification",
        model=SENTIMENT_MODEL,
        device=-1,
    )


def analyze_sentiment(reviews):
    """Analyze review sentiment using the cached pre-trained model."""
    cleaned_reviews = [str(review).strip() for review in reviews if str(review).strip()]
    if not cleaned_reviews:
        return []

    sentiment_pipeline = load_sentiment_pipeline()
    return sentiment_pipeline(
        cleaned_reviews,
        batch_size=8,
        truncation=True,
        max_length=512,
    )


def normalize_sentiment_label(label):
    """Convert model-specific labels into dashboard labels."""
    label_mapping = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive",
        "NEGATIVE": "Negative",
        "NEUTRAL": "Neutral",
        "POSITIVE": "Positive",
    }
    normalized_label = str(label).strip().upper()
    return label_mapping.get(normalized_label, str(label).title())


def forecast_discounts_arima(data, future_days=5):
    """Forecast discounts after converting observations to a daily series."""
    if "Discount" not in data.columns:
        raise ValueError("The competitor data does not contain a Discount column.")

    discount_series = pd.to_numeric(data["Discount"], errors="coerce")
    discount_series.index = pd.to_datetime(discount_series.index, errors="coerce")
    discount_series = discount_series[
        discount_series.index.notna() & discount_series.notna()
    ]

    # Source records contain one observation per day at varying times. Normalize
    # those timestamps and aggregate duplicate calendar dates before forecasting.
    discount_series.index = discount_series.index.normalize()
    discount_series = discount_series.groupby(level=0).mean().sort_index()

    if len(discount_series) < MIN_FORECAST_OBSERVATIONS:
        raise ValueError(
            "At least "
            f"{MIN_FORECAST_OBSERVATIONS} daily observations are required; "
            f"found {len(discount_series)}."
        )

    daily_index = pd.date_range(
        start=discount_series.index.min(),
        end=discount_series.index.max(),
        freq="D",
    )
    discount_series = discount_series.reindex(daily_index)
    discount_series = discount_series.interpolate(method="time").ffill().bfill()
    discount_series.index.name = "Date"

    model = ARIMA(discount_series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=future_days)

    future_dates = pd.date_range(
        start=discount_series.index[-1] + pd.Timedelta(days=1),
        periods=future_days,
        freq="D",
    )
    forecast_values = pd.Series(forecast.to_numpy(), index=future_dates).clip(0, 100)
    return forecast_values.rename("Predicted_Discount").to_frame().rename_axis("Date")


def is_valid_webhook_url(webhook_url):
    """Return whether a configured webhook is a usable HTTPS URL."""
    if not webhook_url:
        return False

    parsed_url = urlparse(webhook_url)
    return parsed_url.scheme == "https" and bool(parsed_url.netloc)


def send_to_slack(message, webhook_url):
    """Send a message to Slack via a validated webhook."""
    response = requests.post(
        webhook_url,
        json={"text": message},
        timeout=15,
    )
    response.raise_for_status()


def prepare_competitor_data(data):
    """Validate and normalize competitor data without deprecated mutations."""
    required_columns = {"Date", "Product_Name", "Price", "MRP", "Discount"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Products.csv is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    prepared_data = data.copy()
    for column in ("Price", "MRP", "Discount"):
        prepared_data[column] = pd.to_numeric(prepared_data[column], errors="coerce")

    prepared_data["Date"] = pd.to_datetime(prepared_data["Date"], errors="coerce")
    prepared_data = prepared_data.dropna(
        subset=["Date", "Product_Name", "Price", "MRP", "Discount"]
    )
    return prepared_data.set_index("Date").sort_index()


def prepare_reviews_data(data):
    """Validate and normalize review data without deprecated mutations."""
    required_columns = {
        "Product_Name",
        "Review_Date",
        "Rating",
        "Review_Title",
        "Review_Text",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Reviews.csv is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    prepared_data = data.copy()
    prepared_data["Review_Date"] = pd.to_datetime(
        prepared_data["Review_Date"], errors="coerce"
    )
    prepared_data["Rating"] = pd.to_numeric(prepared_data["Rating"], errors="coerce")
    return prepared_data.dropna(
        subset=["Product_Name", "Review_Date", "Rating", "Review_Text"]
    )


def summarize_competitor_data(product_data):
    """Create a compact recent-data snapshot suitable for the Groq prompt."""
    snapshot = product_data.reset_index().sort_values("Date").tail(30)
    summary_columns = [
        column
        for column in (
            "Date",
            "Brand",
            "Price",
            "MRP",
            "Discount",
            "Stock_Status",
            "Rating",
        )
        if column in snapshot.columns
    ]
    return snapshot[summary_columns].to_csv(index=False)


def summarize_sentiment(sentiments):
    """Summarize sentiment labels and confidence scores for the Groq prompt."""
    sentiment_data = pd.DataFrame(sentiments)
    if sentiment_data.empty or "label" not in sentiment_data.columns:
        return "No sentiment results available"

    sentiment_data["label"] = sentiment_data["label"].apply(
        normalize_sentiment_label
    )
    sentiment_data["score"] = pd.to_numeric(
        sentiment_data.get("score"), errors="coerce"
    )
    summary = sentiment_data.groupby("label", dropna=False).agg(
        Reviews=("label", "size"),
        Average_Confidence=("score", "mean"),
    )
    return summary.round(3).to_string()


def raise_for_groq_status(response):
    """Raise an HTTP error that includes Groq's useful response message."""
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        try:
            response_data = response.json()
        except ValueError:
            error_detail = response.text.strip()
        else:
            groq_error = (
                response_data.get("error")
                if isinstance(response_data, dict)
                else None
            )
            if isinstance(groq_error, dict):
                error_detail = groq_error.get("message") or groq_error.get("code", "")
            else:
                error_detail = str(groq_error or "")

        message_parts = [str(error)]
        if error_detail:
            message_parts.append(f"Groq API message: {error_detail}")
        if response.status_code == 404:
            message_parts.append(
                "Verify GROQ_MODEL against https://console.groq.com/docs/models."
            )

        raise requests.HTTPError(
            " ".join(message_parts),
            request=error.request,
            response=response,
        ) from error


def generate_strategy_recommendation(
    product_name,
    product_data,
    sentiments,
    api_key,
    model_name=DEFAULT_GROQ_MODEL,
):
    """Generate strategic recommendations with the configured Groq model."""
    if product_data is None or product_data.empty or not sentiments:
        raise ValueError("Competitor and sentiment data are required.")

    competitor_summary = summarize_competitor_data(product_data)
    sentiment_summary = summarize_sentiment(sentiments)

    prompt = f"""
        You are a highly skilled business strategist specializing in **Indian e-commerce**. Based on the following details, suggest strategic recommendations:

        ### **Product Details**
        - **Product Name**: {product_name}
        - **Recent Competitor Data** (Prices are in INR ₹, NOT in USD $):
        {competitor_summary}
        - **Sentiment Analysis Summary**:
        {sentiment_summary}
        - **Today's Date**: {datetime.now().date().isoformat()}

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
    groq_data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model_name,
        "temperature": 0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    groq_response = requests.post(
        GROQ_ENDPOINT,
        json=groq_data,
        headers=headers,
        timeout=60,
    )
    raise_for_groq_status(groq_response)
    return groq_response.json()["choices"][0]["message"]["content"]


############# APP STARTS HERE #############


def main():
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="MarketSense: E-Commerce Competitor Strategy Dashboard",
        page_icon="https://iili.io/2QnnboB.md.jpg",
        layout="wide",
    )

    competitor_data = load_csv("Products.csv")
    reviews_data = load_csv("Reviews.csv")

    try:
        if not competitor_data.empty:
            competitor_data = prepare_competitor_data(competitor_data)
        if not reviews_data.empty:
            reviews_data = prepare_reviews_data(reviews_data)
    except ValueError as error:
        st.error(f"Data validation failed: {error}")
        return

    st.markdown(
        "<h1 style='text-align: center;'>"
        "E-Commerce 🛒 Competitor Strategy Dashboard"
        "</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='text-align: center;'>📌 Select a Product to Analyze 📌</h3>",
        unsafe_allow_html=True,
    )

    products = (
        reviews_data["Product_Name"].dropna().unique().tolist()
        if not reviews_data.empty
        else []
    )
    with st.container():
        _, product_column, _ = st.columns([1, 5, 1])
        with product_column:
            selected_product = (
                st.selectbox(
                    "Product",
                    products,
                    key="product_select",
                    label_visibility="collapsed",
                )
                if products
                else None
            )

    if not selected_product:
        st.warning("⚠️ No products with review data are available.")
        return

    product_reviews = reviews_data[
        reviews_data["Product_Name"] == selected_product
    ]
    product_data = (
        competitor_data[competitor_data["Product_Name"] == selected_product]
        if not competitor_data.empty
        else pd.DataFrame()
    )
    sentiments = []

    st.divider()
    st.header("📈 COMPETITOR ANALYSIS")
    st.subheader(f"📊 Product: {selected_product}")

    if not product_data.empty and "Product_ASIN" in product_data.columns:
        asin_values = product_data["Product_ASIN"].dropna()
        if not asin_values.empty:
            product_image_url = (
                f"https://images.amazon.com/images/P/{asin_values.iloc[0]}.jpg"
            )
            st.markdown(
                "<div style='text-align: center;'>"
                f"<img src='{product_image_url}' alt='Selected product' width='400'/>"
                "</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("🗣️ CUSTOMER SENTIMENT ANALYSIS")

    if product_reviews.empty:
        st.warning("⚠️ No reviews are available for this product.")
    else:
        reviews = product_reviews["Review_Text"].astype(str).tolist()
        try:
            with st.spinner("Analyzing sentiment..."):
                sentiments = analyze_sentiment(reviews)
        except Exception as error:
            st.error(f"Sentiment analysis failed: {error}")

        if sentiments:
            sentiment_df = pd.DataFrame(sentiments)
            sentiment_df["label"] = sentiment_df["label"].apply(
                normalize_sentiment_label
            )
            sentiment_counts = (
                sentiment_df["label"]
                .value_counts()
                .rename_axis("Sentiment")
                .reset_index(name="Reviews")
            )
            color_map = {
                "Negative": "#f54248",
                "Neutral": "#42ddf5",
                "Positive": "#69f542",
            }
            sentiment_chart = px.bar(
                sentiment_counts,
                x="Sentiment",
                y="Reviews",
                color="Sentiment",
                color_discrete_map=color_map,
            )
            st.plotly_chart(sentiment_chart, width="stretch")

            reviews_to_display = product_reviews[
                ["Review_Title", "Review_Text"]
            ].head(10).copy()
            reviews_to_display.index = range(1, len(reviews_to_display) + 1)
            st.dataframe(reviews_to_display, width="stretch")

    st.divider()
    if product_data.empty:
        st.warning("⚠️ No competitor history is available for this product.")
    else:
        try:
            with st.spinner("Forecasting discounts using ARIMA..."):
                forecast_df = forecast_discounts_arima(product_data, future_days=5)

            st.subheader("❄️ FORECASTED DISCOUNTS")
            st.line_chart(forecast_df["Predicted_Discount"], width="stretch")
            st.dataframe(forecast_df, width="stretch")
        except Exception as error:
            st.error(f"ARIMA forecasting failed: {error}")

    st.divider()
    st.subheader("📌 STRATEGIC RECOMMENDATIONS 🧠 🌌")

    groq_api_key = get_config_value("GROQ_API_KEY")
    if product_data.empty or not sentiments:
        st.info(
            "Recommendations require both competitor history and sentiment results."
        )
    elif not groq_api_key:
        st.info(
            "Groq recommendations are disabled. Set the GROQ_API_KEY environment "
            "variable and restart the app to enable them."
        )
    else:
        try:
            with st.spinner("Generating strategic recommendations..."):
                response = generate_strategy_recommendation(
                    product_name=selected_product,
                    product_data=product_data,
                    sentiments=sentiments,
                    api_key=groq_api_key,
                    model_name=get_config_value(
                        "GROQ_MODEL", DEFAULT_GROQ_MODEL
                    ),
                )
            st.success(response)

            slack_webhook = get_config_value("SLACK_WEBHOOK_URL")
            if is_valid_webhook_url(slack_webhook):
                if st.button("Send recommendation to Slack"):
                    try:
                        with st.spinner("Sending recommendation to Slack..."):
                            send_to_slack(response, slack_webhook)
                        st.success("Recommendation sent to Slack.")
                    except requests.RequestException as error:
                        st.error(f"Failed to send the Slack notification: {error}")
            elif slack_webhook:
                st.warning(
                    "Slack notification is disabled because SLACK_WEBHOOK_URL "
                    "is not a valid HTTPS URL."
                )
            else:
                st.caption(
                    "Slack notification is optional. Set SLACK_WEBHOOK_URL to "
                    "enable it."
                )
        except requests.RequestException as error:
            st.error(f"Groq request failed: {error}")
        except (KeyError, TypeError, ValueError) as error:
            st.error(f"Groq returned an unexpected response: {error}")

    st.markdown(
        "<h6 style='text-align: center;'>"
        f"© {datetime.now().year} MarketSense: E-Commerce Strategy Dashboard"
        "</h6>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
