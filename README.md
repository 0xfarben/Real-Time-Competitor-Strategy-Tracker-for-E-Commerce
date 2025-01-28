# Real-Time Competitor Strategy Tracker for E-Commerce  

## Project Overview  
This project provides real-time competitive intelligence for e-commerce businesses. It tracks competitor pricing, discount strategies, and customer sentiment using machine learning and large language models.  
<div align="center">
  <img src="favicon.jpeg" alt="Centered Image" width="400">
</div>

## Demonstration Video

```
https://youtu.be/watch?v=C1RlDA-2-mY
```
<div style="text-align: center;">
``` <a href="https://youtu.be/watch?v=C1RlDA-2-mY"><img src="http://img.youtube.com/vi/C1RlDA-2-mY/0.jpg" alt="My Awesome Video"></a>```
</div>

## Features  
- **Competitor Data Aggregation**: Track pricing and discount strategies.  
- **Sentiment Analysis**: Analyze customer reviews for actionable insights.  
- **Predictive Modeling**: Forecast competitor discounts.  
- **Slack Integration**: Get real-time notifications on competitor activity.  

## Setup Instructions  

### 1. Clone the Repository  
```sh  
git clone <repository-url>  
cd <repository-directory>  
```

### 2. Install Dependencies  
```sh  
pip install -r requirements.txt  
```

### 3. Configure API Keys  

#### Groq API Key  
1. Sign up for a Groq account at [Groq](https://groq.com).  
2. Obtain your API key from the Groq dashboard.  
3. Add the API key to the `app.py` file.  

#### Slack Webhook Integration  
1. Go to the [Slack API](https://api.slack.com).  
2. Create a new app and enable **Incoming Webhooks**.  
3. Add a webhook to a Slack channel and copy the generated URL.  
4. Add this URL to the `app.py` file.  

### 4. Run the Application  
```sh  
streamlit run app.py  
```

## Project Files  
- `app.py` - Main application script.
- `get_products.py` - Script to fetch the products.
- `get_reviews_data.py` - Script to fetch the product's review data.
- `reviews.csv` - Sample reviews data for sentiment analysis.  
- `products.csv` - Sample competitor data for analysis. (https://drive.google.com/file/d/1sL92n8450F_bpNQkwqb-7FUlDtx0tN-P/view?usp=sharing)
- `requirements.txt` - List of dependencies.  

## Usage  
1. Launch the Streamlit app.  
2. Select a product from the dropdown menu.  
3. View competitor analysis, sentiment trends, and discount forecasts.  
4. Get strategic recommendations and real-time notifications.  

## License  
This project is licensed under the MIT License.
