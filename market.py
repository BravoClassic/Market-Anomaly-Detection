# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# %%
# Look through the market files and store in dataframe

with open("/content/Financial_Market_Data.csv", "r") as f:
  df = pd.read_csv(f)

df.describe()

# %%
df.info()

# %%
sns.set_style('whitegrid')
plt.figure(figsize=(12,10))

# %%
sns.countplot(x='Y', data=df)
plt.title("Distribution of Market Crash") # 0 means no crash and 1 means crush


# %%
sns.histplot(data=df, x='GBP', kde=True)
plt.title("GBP Distribution")

# %%
sns.histplot(data=df, x='JPY', kde=True)
plt.title("JPY Distribution")

# %%
sns.histplot(data=df, x='USGG30YR', kde=True)
plt.title("USGG30YR Distribution")

# %%
features = df.drop('Y', axis=1)
features

# %%
target = df['Y']
target

# %%
features = features.drop(['Data'], axis=1)
features

# %%
# Handle missing values
features = features.dropna()
features

# %%
from sklearn.model_selection import train_test_split
# scaling features to enable values there have high values then normal doesn't affect the model's learning process.
from sklearn.preprocessing import StandardScaler

# %%
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, train_size=0.8, random_state=42)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# %%
def evaluate_and_save_model(model, x_train, x_test, y_train, y_test, filename):
  model.fit(x_train, y_train)
  y_predictions = model.predict(x_test)
  accuracy = accuracy_score(y_test, y_predictions)
  print(f"{model.__class__.__name__} Accuracy: {accuracy:.4f}")
  print(f"\nClassification Report:\n{classification_report(y_test, y_predictions)}")
  print("-----------------")
  with open(filename, 'wb') as file:
    pickle.dump(model,file)
  print(f"Model saved as {filename}\n")

# %%
lr_model = LogisticRegression(random_state=42)
lr_model.fit(x_train, y_train)
lr_predictions = lr_model.predict(x_test)

# %%
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_accuracy

# %%
x_gbm_model = XGBClassifier(random_state=42)
evaluate_and_save_model(x_gbm_model, x_train, x_test, y_train, y_test, "xgb_model.pkl")

# %%
# Using the xgboost model
feature_importance = x_gbm_model.feature_importances_
feature_names = features.columns

# %%
feature_importance_df = pd.DataFrame(
    {
        'Feature': feature_names,
        'Importance': feature_importance
        })
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# %%
feature_importance_df

# %%
plt.figure(figsize=(10,6))
sns.barplot(x='Feature', y='Importance', data=feature_importance_df)
plt.xticks(rotation=90)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# %%
features = df[['VIX','GTITL30YR',"GTITL2YR",'GTITL10YR', 'GTDEM30Y',"EONIA", 'GTDEM10Y', 'GTDEM2Y', 'DXY', 'GTJPY2YR',"GTGBP30Y"]]

# %%
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, train_size=0.8, random_state=42)

# %%
lr_model = LogisticRegression(random_state=42)
lr_model.fit(x_train, y_train)
lr_predictions = lr_model.predict(x_test)

# %%
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_accuracy

# %%
x_gbm_model = XGBClassifier(random_state=42)
evaluate_and_save_model(x_gbm_model, x_train, x_test, y_train, y_test, "xgb_model_1.pkl")

# %%
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

y_proba = x_gbm_model.predict_proba(x_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Plot Precision-Recall Curve
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Area Under PR Curve
pr_auc = auc(recall, precision)
print(f'PR AUC: {pr_auc}')


# %%
# Correlation Matrix
plt.figure(figsize=(52, 38))
correlation_matrix = df.drop(['Data'],axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()




# %%
from scipy.stats import ttest_ind

# Testing significance of a numerical feature
tickers = ['VIX','GTITL30YR',"GTITL2YR",'GTITL10YR', 'GTDEM30Y',"EONIA", 'GTDEM10Y', 'GTDEM2Y', 'DXY', 'GTJPY2YR',"GTGBP30Y"]

ticker_dict = {}

# Perform t-test
for ticker in tickers:
  for other_ticker in tickers:
    if ticker != other_ticker:
      group1 = df[df['Y'] == 0][ticker]
      group2 = df[df['Y'] == 1][other_ticker]

      t_stat, p_val = ttest_ind(group1, group2)
      # print(f"T-Test Statistic: {t_stat}, p-value: {p_val}")

      if p_val < 0.05:
          # print("Feature is statistically significant.")
          ticker_dict[ticker+"-"+other_ticker] = "Feature is statistically significant."
      else:
          # print("Feature is not statistically significant.")
          ticker_dict[ticker+"-"+other_ticker] = "Feature is not statistically significant."
for ticker in ticker_dict:
  print(ticker+": ", ticker_dict[ticker])


# %%
# Example usage
# portfolio = {
#     'VIX': 16.13,
#     'DXY': 103.25,
#     'EONIA': -0.50,
#     'GTITL30YR': 2.75,
#     'GTITL10YR': 1.50,
#     'GTITL2YR': 0.25,
#     'GTDEM30Y': 0.75,
#     'GTDEM10Y': -0.10,
#     'GTDEM2Y': -0.75,
#     'GTJPY2YR': -0.15,
#     'GTGBP30Y': 1.25
# }

# 2021
portfolio_2 = {
    'VIX': 33.09,
    'DXY': 90.24,
    'EONIA': -0.48,
    'GTITL30YR': 1.42,
    'GTITL10YR': 0.77,
    'GTITL2YR': -0.39,
    'GTDEM30Y': 0.205,
    'GTDEM10Y': -0.444,
    'GTDEM2Y': 8.426,
    'GTJPY2YR': -0.125,
    'GTGBP30Y': 1.108
}

portfolio = [16.13, 103.25, -0.50, 2.75, 1.50, 0.25, 0.75, -0.10, -0.75, -0.15, 1.25]
portfolio = pd.DataFrame([portfolio], columns=['VIX','GTITL30YR',"GTITL2YR",'GTITL10YR', 'GTDEM30Y',"EONIA", 'GTDEM10Y', 'GTDEM2Y', 'DXY', 'GTJPY2YR',"GTGBP30Y"]) # Add column names here
prediction = x_gbm_model.predict(portfolio)  # 0 or 1
proba_prediction = x_gbm_model.predict_proba(portfolio)
print("2025 market crash prediction",prediction[0])
print("2025 market crash prediction probability",proba_prediction)

port = []
for key in portfolio_2:
  port.append(portfolio_2[key])
portf = pd.DataFrame([port], columns=['VIX','GTITL30YR',"GTITL2YR",'GTITL10YR', 'GTDEM30Y',"EONIA", 'GTDEM10Y', 'GTDEM2Y', 'DXY', 'GTJPY2YR',"GTGBP30Y"]) # Add column names here
prediction = x_gbm_model.predict(portf)  # 0 or 1
proba_prediction = x_gbm_model.predict_proba(portf)
print("2021 market crash prediction",prediction[0])
print("2021 market crash prediction probability",proba_prediction)

# %%
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(x_train, y_train)


# %%
# Train XGBoost
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eta=0.1,
    max_depth=5,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=1000,
    scale_pos_weight=177/46
)

xgb_model.fit(X_resampled, y_resampled)

# Predict with adjusted threshold
y_pred_proba = xgb_model.predict_proba(x_test)[:, 1]
threshold = 0.1
y_pred = (y_pred_proba >= threshold).astype(int)

# Evaluate
print(classification_report(y_test, y_pred))

# %%
portfolio = [16.13, 103.25, -0.50, 2.75, 1.50, 0.25, 0.75, -0.10, -0.75, -0.15, 1.25]
portfolio = pd.DataFrame([portfolio], columns=['VIX','GTITL30YR',"GTITL2YR",'GTITL10YR', 'GTDEM30Y',"EONIA", 'GTDEM10Y', 'GTDEM2Y', 'DXY', 'GTJPY2YR',"GTGBP30Y"]) # Add column names here
prediction = xgb_model.predict(portfolio)  # 0 or 1
proba_prediction = xgb_model.predict_proba(portfolio)
print("2025 market crash prediction",prediction[0])
print("2025 market crash prediction probability",proba_prediction)

# %%

# 2021
portfolio_2 = {
    'VIX': 33.09,
    'DXY': 90.24,
    'EONIA': -0.48,
    'GTITL30YR': 1.42,
    'GTITL10YR': 0.77,
    'GTITL2YR': -0.39,
    'GTDEM30Y': 0.205,
    'GTDEM10Y': -0.444,
    'GTDEM2Y': 8.426,
    'GTJPY2YR': -0.125,
    'GTGBP30Y': 1.108
}

port = []
for key in portfolio_2:
  port.append(portfolio_2[key])
portf = pd.DataFrame([port], columns=['VIX','GTITL30YR',"GTITL2YR",'GTITL10YR', 'GTDEM30Y',"EONIA", 'GTDEM10Y', 'GTDEM2Y', 'DXY', 'GTJPY2YR',"GTGBP30Y"]) # Add column names here
prediction = xgb_model.predict(portf)  # 0 or 1
proba_prediction = xgb_model.predict_proba(portf)
print("2021 market crash prediction",prediction[0])
print("2021 market crash prediction probability",proba_prediction)

# %% [markdown]
# Part 2: Streamlit Web app

# %%
!pip install streamlit pyngrok python-dotenv openai groq yfinance investpy ffn

# %%

import yfinance as yf
# VIX','GTITL30YR',"GTITL2YR",'GTITL10YR', 'GTDEM30Y',"EONIA", 'GTDEM10Y', 'GTDEM2Y', 'DXY', 'GTJPY2YR',"GTGBP30Y"
tickers = yf.Tickers('USDX VIX 0DMT.IL')
tickers.tickers['0DMT.IL'].history(period="max")

# %%

# # Loop through the bonds to fetch historical data
# for bond in bonds:
#     try:
#         bond_data = investpy.bonds.get_bonds_overview(country=bond["country"],as_json=True)
#         for single_bond in bond_data:
#           if single_bond["name"] == bond["name"]:
#             print(single_bond)
#     except Exception as e:
#         print(f"Failed to fetch data for {bond['name']}: {e}")
# try:
#   indices = investpy.indices.get_indices(country="united states")
#   # print(indices)
#   print(indices.loc[indices['name'] == 'DXY'])
#   index = investpy.indices.get_index_historical_data(index="NQ US Mid Cap Value",country="united states",from_date="01/01/2024",to_date="01/01/2025")
#   # index = investpy.indices.get_index_historical_data(index="DXY",country="united states",interval="daily")
#   index.head()
# except Exception as e:
#         print(f"Failed to fetch data for: {e}")

# %%
import requests

url_search = "https://iappapi.investing.com/search.php"
headers = {
    "X-Meta-Ver": "14"
}
params = {
    "string": "DXY"
}

data = requests.get(url_search, headers=headers, params=params)

# Print the response content
print(data.text)

# %%
import datetime
from zoneinfo import ZoneInfo
def date_to_unix(date_str):
    """
    Convert a date string in DDMMYYYY format at 00:00 to a Unix timestamp.

    Args:
        date_str (str): The date string in DDMMYYYY format (e.g., "23092024").

    Returns:
        int: The Unix timestamp at 00:00 of the given date.
    """
    # Parse the date string in DDMMYYYY format
    date_obj = datetime.datetime.strptime(date_str, "%d%m%Y")
    local_tz = ZoneInfo("localtime")
    # Convert to Unix timestamp
    unix_timestamp = int(date_obj.replace(tzinfo=local_tz).timestamp())

    return unix_timestamp

# %%
from tabulate import tabulate

# # API endpoint
url_get_data = "https://iappapi.investing.com/get_screen.php"

# use date_to as the date specified from user to get the data
def get_bond_data(date_to="28092024", date_from="23092024"):
  # # Define countries and maturities
  bonds = [
      {"country": "Italy", "name": "Italy 30Y", "close":0},
      {"country": "Italy", "name": "Italy 2Y","close":0},
      {"country": "Italy", "name": "Italy 10Y","close":0},
      {"country": "Germany", "name": "Germany 30Y","close":0},
      {"country": "Germany", "name": "Germany 10Y","close":0},
      {"country": "Germany", "name": "Germany 2Y","close":0},
      {"country": "Japan", "name": "Japan 2Y","close":0},
      {"country": "United Kingdom", "name": "U.K. 30Y","close":0},
      {"country":"United States", "name":"DXY","close":0},
      {"country":"United States", "name":"VIX","close":0},
      {"country":"United States", "name":"Euro Short-Term Rate Futures","close":0}

  ]


  try:
    for key, bond in enumerate(bonds):
      parameter = {
          "string": bond["name"]
      }
      bond_req = requests.get(url_search, headers=headers, params=parameter)
      ticker_name = bond_req.json()["data"]["pairs_attr"][0]["search_main_longtext"]
      ticker_id = bond_req.json()["data"]["pairs_attr"][0]["pair_ID"]
      print(f"{ticker_name}: {ticker_id}")
      parameters = {
      "lang_ID": "51",
      "skinID": "2",
      "interval": "day",
      "time_utc_offset": "0",
      "screen_ID": "63",
      "pair_ID": bond_req.json()["data"]["pairs_attr"][0]["pair_ID"],
      "date_to": date_to,
      "date_from": date_from
      }
      bond_data = requests.get(url_get_data, headers=headers, params=parameters)
      print(bond_data.json())
      unix = date_to_unix(date_to)
      print(unix)

      for result in bond_data.json()["data"]:
        for i in range(len(result["screen_data"]["data"])):
          if result["screen_data"]["data"][i]["date"] == unix:
            bonds[key]["close"] = result["screen_data"]["data"][i]["close"]
            print(result["screen_data"]["data"][i]["date"])
            print(result["screen_data"]["data"][i]["close"])
        print()
        print(tabulate(result["screen_data"]["data"],headers="keys"))
        print("----------------------------------------------------------------------------------------------------------------")
  except Exception as e:
          print(f"Failed to fetch data for: {e}")
  print(bonds)
get_bond_data()


# %%


# # API endpoint
url = "https://iappapi.investing.com/get_screen.php"

# # Headers (check for any required authentication headers)
headers = {
    "X-Meta-Ver": "14"
}

# Parameters (replace with actual required values)
params = {
    "lang_ID": "51",
    "skinID": "2",
    "interval": "day",
    "time_utc_offset": "0",
    "screen_ID": "63",
    "pair_ID": data.json()["data"]["pairs_attr"][0]["pair_ID"],
    "date_to": "28092024",
    "date_from": "23092024"
}

try:
    # Send GET request with a timeout
    response = requests.get(url, headers=headers, params=params, timeout=10)

    # Check for HTTP response errors
    response.raise_for_status()

    # Print the response content
    print("Response Status Code:", response.status_code)
    print("Response JSON:", response.json())  # Assuming the response is JSON
except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
except requests.exceptions.RequestException as req_err:
    print(f"Request error occurred: {req_err}")


# %%
from sklearn.ensemble import IsolationForest
forest = IsolationForest(random_state=42)
# x_train, x_test, y_train, y_test
forest.fit(x_train, y_train)

# %%
forest_pred = forest.predict(x_test)

# %%
lr_accuracy = accuracy_score(y_test, forest_pred)
lr_accuracy

# %%
evaluate_and_save_model(forest, X_resampled, x_test, y_resampled, y_test, "forest.pkl")
#X_resampled, y_resampled

# %%
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
clf = make_pipeline(StandardScaler(), SVC(gamma='scale'))
# clf.fit(x_train, y_train)
evaluate_and_save_model(clf, x_train, x_test, y_train, y_test, "svm.pkl")
# y_resampled


# %%
portfolio = [16.13, 103.25, -0.50, 2.75, 1.50, 0.25, 0.75, -0.10, -0.75, -0.15, 1.25]
portfolio = pd.DataFrame([portfolio], columns=['VIX','GTITL30YR',"GTITL2YR",'GTITL10YR', 'GTDEM30Y',"EONIA", 'GTDEM10Y', 'GTDEM2Y', 'DXY', 'GTJPY2YR',"GTGBP30Y"]) # Add column names here
prediction = clf.predict(portfolio)  # 0 or 1
# proba_prediction = clf.predict_proba(portfolio)
print("2025 market crash prediction",prediction[0])
# print("2025 market crash prediction probability",proba_prediction)

# %%

# 2021
portfolio_2 = {
    'VIX': 33.09,
    'DXY': 90.24,
    'EONIA': -0.48,
    'GTITL30YR': 1.42,
    'GTITL10YR': 0.77,
    'GTITL2YR': -0.39,
    'GTDEM30Y': 0.205,
    'GTDEM10Y': -0.444,
    'GTDEM2Y': 8.426,
    'GTJPY2YR': -0.125,
    'GTGBP30Y': 1.108
}

port = []
for key in portfolio_2:
  port.append(portfolio_2[key])
portf = pd.DataFrame([port], columns=['VIX','GTITL30YR',"GTITL2YR",'GTITL10YR', 'GTDEM30Y',"EONIA", 'GTDEM10Y', 'GTDEM2Y', 'DXY', 'GTJPY2YR',"GTGBP30Y"]) # Add column names here
prediction = clf.predict(portf)  # 0 or 1
# proba_prediction = clf.predict_proba(portf)
print("2021 market crash prediction",prediction[0])
# print("2021 market crash prediction probability",proba_prediction)

# %%
from threading import Thread
from pyngrok import ngrok
import streamlit as st
from google.colab import userdata
import os


# %%
authtoken = userdata.get('NGROK_AUTH_TOKEN')
ngrok.set_auth_token(authtoken)

# %%
def run_streamlit():
  os.system('streamlit run /content/app.py --server.port 8501')

# %%
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
os.environ["GROK_API_KEY"] = userdata.get('GROK_API_KEY')
os.environ["FINANCE_API"] = userdata.get('FINANCE_API')

# %%
import requests
import os

api_key = os.environ.get("GROK_API_KEY")
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)


# %%
# https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=VXX&interval=5min&month=2019-01&outputsize=full&apikey=8H1H25B6U5UNZ7UK
#For VIX,

# https://api.estr.dev/latest
# https://api.estr.dev/historical
#

# %%
%%writefile app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np
import plotly.graph_objects as go
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from google.colab import userdata
import seaborn as sns
from pyngrok import ngrok
from dotenv import load_dotenv
import os
import google.generativeai as genai
from openai import OpenAI
from groq import Groq
import pickle
import pandas as pd
import requests
import datetime


load_dotenv()

genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))


def date_to_unix(date_str):
    """
    Convert a date string in DDMMYYYY format at 00:00 to a Unix timestamp.

    Args:
        date_str (str): The date string in DDMMYYYY format (e.g., "23092024").

    Returns:
        int: The Unix timestamp at 00:00 of the given date.
    """
    # Parse the date string in DDMMYYYY format
    date_obj = datetime.datetime.strptime(date_str, "%d%m%Y")

    # Convert to Unix timestamp
    unix_timestamp = int(date_obj.replace(tzinfo=datetime.timezone.utc).timestamp())

    return unix_timestamp

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
def prompt_text(model_prediction, confidence):
    prompt = f"""You are a financial advisor specializing in data-driven investment strategies.
    You are tasked with explaining an AI-driven investment recommendation to an end user.

    The AI model predicted that the market is in a state of '{model_prediction}' with a confidence of {confidence*100}%.

    Based on this prediction, the investment strategy recommends:
    - Adjusting the portfolio allocation to minimize risk or maximize returns.
    - Shifting investment focus based on market conditions (e.g., increase allocation to safe-haven assets like gold during a crash).

    In your response:
    - Clearly explain what the prediction means for the user's portfolio.
    - Describe why the model might have made this prediction based on market trends or indicators.
    - Provide actionable steps the user can take to align with the strategy.
    - Avoid repeating phrases like 'The AI model predicted the market state is...' in your explanation.
    - Keep your explanation concise (4 sentences max).

    Let's think step by step about this. Verify step by step your explanation for accuracy and clarity.
    """

    return prompt

def load_model(model_path):
  # Load the model
  with open(model_path, 'rb') as f:
      loaded_model = pickle.load(f)
  return loaded_model

def generate_explanation(model_prediction, confidence,llm_model):
  prompt = prompt_text(model_prediction, confidence)

  if llm_model == "OpenAI":

    # Getting the base64 string
    client = OpenAI(
        api_key=OPENAI_API_KEY
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt,
            },
          ],
        }
      ],
    )
    return response.choices[0].message.content
  elif llm_model == "Google":
    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
    response = model.generate_content([prompt])
    return response.text
  elif llm_model == "Llama Vision":
    client = Groq(api_key=os.environ.get("GROK_API_KEY"))

    response = client.chat.completions.create(
      model="llama-3.2-11b-vision-preview",
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt,
            },

          ]

        }
      ],
    )
    return response.choices[0].message.content


st.title('Market Anomaly Detection')
st.write('Enter values for these tickers to predict market crash. You either enter all the values manually')
model = load_model('xgb_model_1.pkl')

# option = st.selectbox(
#     "Select Data Options",
#     ("Manual values", "API - Alpha Vantage"),
# )

# if option == "Manual values":
col1, col2 = st.columns(2)
# 'VIX','GTITL30YR',"GTITL2YR",'GTITL10YR', 'GTDEM30Y',"EONIA", 'GTDEM10Y', 'GTDEM2Y', 'DXY', 'GTJPY2YR',"GTGBP30Y"
#  'VXX',
with col1:
  VIX = st.number_input("VIX", format="%0.3f")
  GTITL30YR = st.number_input("GTITL30YR", format="%0.3f")
  GTITL2YR = st.number_input("GTITL2YR", format="%0.3f")
  GTITL10YR = st.number_input("GTITL10YR", format="%0.3f")
  GTDEM30Y = st.number_input("GTDEM30Y", format="%0.3f")
  EONIA = st.number_input("EONIA", format="%0.3f")
with col2:
  GTDEM10Y = st.number_input("GTDEM10Y", format="%0.3f")
  GTDEM2Y = st.number_input("GTDEM2Y", format="%0.3f")
  DXY = st.number_input("DXY", format="%0.3f")
  GTJPY2YR = st.number_input("GTJPY2YR", format="%0.3f")
  GTGBP30Y = st.number_input("GTGBP30Y", format="%0.3f")

# st.session_state["prediction"] = ""
# st.button("Reset", type="primary")
# if st.button("Predict"):
llm = st.selectbox(
    "Select LLM",
    ("OpenAI", "Google")
  )
portf = pd.DataFrame([[VIX, GTITL30YR, GTITL2YR , GTITL10YR, GTDEM30Y, EONIA, GTDEM10Y, GTDEM2Y, DXY, GTJPY2YR, GTGBP30Y]], columns=['VIX','GTITL30YR',"GTITL2YR",'GTITL10YR', 'GTDEM30Y',"EONIA", 'GTDEM10Y', 'GTDEM2Y', 'DXY', 'GTJPY2YR',"GTGBP30Y"]) # Add column names here

predicted = model.predict(portf)
proba_prediction = model.predict_proba(portf)
result = "Market Crash" if predicted[0] == 1 else "No Market Crash"
st.metric(label="Market Crash Prediction", value=result)
if llm is not None:
  explanation = generate_explanation(result, proba_prediction[0][predicted][0], llm)
  prmpt = prompt_text(result, proba_prediction[0][predicted][0])

st.write(explanation)

# elif option == "API - Finnhub":
#   tickers = ['VIX','GTITL30YR',"GTITL2YR",'GTITL10YR', 'GTDEM30Y',"EONIA", 'GTDEM10Y', 'GTDEM2Y', 'DXY', 'GTJPY2YR',"GTGBP30Y"]
#   market_date = st.date_input("Select Date")
#   if market_date.isoweekday() > 5:
#     st.write("Market is closed on weekends")
#   else:

#     portf = pd.DataFrame([[VIX, GTITL30YR, GTITL2YR , GTITL10YR, GTDEM30Y, EONIA, GTDEM10Y, GTDEM2Y, DXY, GTJPY2YR, GTGBP30Y]], columns=['VIX','GTITL30YR',"GTITL2YR",'GTITL10YR', 'GTDEM30Y',"EONIA", 'GTDEM10Y', 'GTDEM2Y', 'DXY', 'GTJPY2YR',"GTGBP30Y"]) # Add column names here

#     predicted = model.predict(portf)
#     st.write(predicted)
#     proba_prediction = model.predict_proba(portf)
#     result = "Market Crash" if predicted[0] == 1 else "No Market Crash"
#     st.metric(label="Market Crash Prediction", value=result)
#     st.session_state["prediction"] = result
#     st.header("Chat with Investment Expert")
#     client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY') )

#     # Set a default model
#     if "openai_model" not in st.session_state:
#         st.session_state["openai_model"] = "gpt-4o"


#     # Initialize chat history

#     if "messages" not in st.session_state or st.session_state.messages == [] and st.session_state.prediction:
#         st.session_state.messages = [
#             {"role": "user", "content": f"{prmpt}"},
#           {"role": "assistant", "content": explanation}

#         ]

#     # Display chat messages from history on app rerun

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Accept user input
#     if prompt := st.chat_input("Answer any question about current market condition?"):
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": str(prompt)})
#         # Display user message in chat message container
#         with st.chat_message("user"):
#             st.markdown(prompt)
#         # Display assistant response in chat message container
#         with st.chat_message("assistant"):
#           stream = client.chat.completions.create(
#               model=st.session_state["openai_model"],
#               messages=[
#                   {"role": m["role"], "content": m["content"]}
#                   for m in st.session_state.messages
#               ],
#               stream=True,
#           )
#           response = st.write_stream(stream)
#         st.session_state.messages.append({"role": "assistant", "content": response})


# %%
thread = Thread(target=run_streamlit)
thread.start()

# %%
tunnels = ngrok.get_tunnels()
for tunnel in tunnels:
  print(f"Closing tunnel: {tunnel.public_url} -> {tunnel.config['addr']}")
  ngrok.disconnect(tunnel.public_url)

# %%
public_url = ngrok.connect(8501,proto='http', bind_tls=True)
public_url


