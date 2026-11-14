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
 <a href="https://youtu.be/watch?v=C1RlDA-2-mY"><img src="http://img.youtube.com/vi/C1RlDA-2-mY/0.jpg" alt="My Awesome Video"></a>
</div>

## Features  
- **Competitor Data Aggregation**: Track pricing and discount strategies.  
- **Sentiment Analysis**: Analyze customer reviews for actionable insights.  
- **Predictive Modeling**: Forecast competitor discounts.  
- **Slack Integration**: Get real-time notifications on competitor activity.  

## Setup Instructions  

### 1. Install Python 3.12

This project requires **Python 3.12.x**. Other Python versions are not supported and may cause dependency or runtime errors.

Verify that Python 3.12 is installed:

**Windows**
```powershell
py -3.12 --version
```

**macOS/Linux**
```sh
python3.12 --version
```

The command should report `Python 3.12.x` before you continue.

### 2. Clone the Repository
```sh  
git clone <repository-url>  
cd <repository-directory>  
```

### 3. Create a Python 3.12 Virtual Environment and Install Dependencies

**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS/Linux**
```sh  
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After activation, run `python --version` and confirm that it reports `Python 3.12.x`.

### 4. Configure API Keys

#### Groq API Key
1. Sign up for a Groq account at [Groq](https://groq.com).
2. Obtain your API key from the Groq dashboard.
3. Set the key in the terminal where you will launch Streamlit:

**Windows (PowerShell)**
```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

**macOS/Linux**
```sh
export GROQ_API_KEY="your_groq_api_key"
```

The app uses `llama-3.1-8b-instant` by default. To use another available Groq model, set the optional `GROQ_MODEL` environment variable.

#### Slack Webhook Integration (Optional)
1. Go to the [Slack API](https://api.slack.com).
2. Create a new app and enable **Incoming Webhooks**.
3. Add a webhook to a Slack channel and copy the generated HTTPS URL.
4. Set the webhook in the terminal where you will launch Streamlit:

**Windows (PowerShell)**
```powershell
$env:SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

**macOS/Linux**
```sh
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

If this variable is omitted, Slack notifications are disabled without affecting the rest of the dashboard. Do not place API keys or webhook URLs directly in `app.py`.

### 5. Run the Application
```sh  
python -m streamlit run app.py
```

## Project Files  
- `app.py` - Main application script.
- `Get_Products.py` - Script to fetch the products.
- `Get_Reviews_Data.py` - Script to fetch the product's review data.
- `Reviews.csv` - Sample reviews data for sentiment analysis.
- `Products.csv` - Sample competitor data for analysis. (https://drive.google.com/file/d/1sL92n8450F_bpNQkwqb-7FUlDtx0tN-P/view?usp=sharing)
- `requirements.txt` - List of dependencies.  

## Usage  
1. Launch the Streamlit app.  
2. Select a product from the dropdown menu.  
3. View competitor analysis, sentiment trends, and discount forecasts.  
4. Get strategic recommendations and real-time notifications.  

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

