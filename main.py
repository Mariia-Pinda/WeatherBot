import os
import requests
import math
from datetime import *
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import openai
import logging
import joblib

# Required tokens
TELEGRAM_TOKEN = '6566380076:AAEBP-MQ3b7PZ2Fvw8R6lwDafhkXRa4e0eU'
WEATHER_API_TOKEN = '817432b50ab14e23ba4265971961b4c9'
OPENAI_API_KEY = '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'


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        # Define the LSTM layer with dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)

        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.linear(out[:, -1, :])
        return out
# Move the model to the device
input_dim = 14  # number of features
hidden_dim = 64  # number of hidden states in the LSTM
num_layers = 2  # number of LSTM layers
output_dim = 14  # number of predicted features (same as input features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                  dropout_rate=0.3)
model.load_state_dict(torch.load('./data/model_state_dict.pth'))
model.to(device)
model.eval()
openai.api_key = OPENAI_API_KEY

lat = 50.4501
lon = 30.5234
# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_historical_weather(lat, lon, hours, api_key):
    # Create an empty list to store the data
    weather_data = []

    for i in range(hours):
        # Calculate the timestamps for each hour
        end_time = datetime.utcnow() - timedelta(hours=i)
        start_timestamp = int(end_time.timestamp())

        # Build the API URL
        api_url = (
            f"https://api.openweathermap.org/data/3.0/onecall/timemachine?"
            f"lat={lat}&lon={lon}"
            f"&dt={start_timestamp}"
            f"&units=metric"
            f"&appid={api_key}"
        )

        # Make the API call
        response = requests.get(api_url)

        # Check if the API call was successful
        if response.status_code == 200:
            # Extract the data from the response
            data = response.json()['data'][0]
            hour_data = {
                'temp': data['temp'],
                'dew_point': data['dew_point'],
                'feels_like': data['feels_like'],
                'pressure': data['pressure'],
                'humidity': data['humidity'],
                'wind_speed': data['wind_speed'],
                'wind_deg': data['wind_deg'],
                'rain_1h': data.get('rain', {}).get('1h', 0),
                'snow_1h': data.get('snow', {}).get('1h', 0),
                'clouds_all': data['clouds'],
                'hour_sin': math.sin(2 * math.pi * end_time.hour / 24),
                'hour_cos': math.cos(2 * math.pi * end_time.hour / 24),
                'month_sin': math.sin(2 * math.pi * end_time.month / 12),
                'month_cos': math.cos(2 * math.pi * end_time.month / 12)
            }
            print(hour_data)
            # Append this hour's data to the list
            weather_data.append(hour_data)
        else:
            print(f"Error fetching data: {response.json()}")
            return None

    df = pd.DataFrame(weather_data)

    scaler = joblib.load('./data/scaler.gz')
    scaled_data = scaler.transform(df)

    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    print(scaled_df)
    return scaled_df.values.tolist()




async def get_openai_response(update: Update, prompt: str) -> str:
    endpoint = "https://api.openai.com/v1/engines/gpt-3.5/completions"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "A user has asked for a weather forecast at some location(city)."
                                          "Please print exactly one word with the name of the city."
                                          "Print the name of the city in english."
                                          "If there is no location in this text - print null"},
            {"role": "user", "content": prompt}
        ]
    )
    print(response)
    city = response["choices"][0]["message"]["content"].strip()
    print(city)
    weather_response = await get_weather(update, city)
    print(weather_response)
    data = get_historical_weather(lat, lon, 10, WEATHER_API_TOKEN)
    data = torch.tensor(data, dtype=torch.float32).to(device).unsqueeze(0)
    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                      dropout_rate=0.3)

    with torch.no_grad():
        model.load_state_dict(torch.load('./data/model_state_dict.pth'))
        model.to(device)
        model.eval()
        out = model(data).squeeze(0).detach().cpu().numpy()
    print(out)
    scaler = joblib.load('./data/scaler.gz')
    print(scaler.data_min_, scaler.data_max_)
    out = scaler.inverse_transform(out.reshape(1, -1))
    print(out)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that wants to help with the weather."
                                          "You will get a users prompt, data from my model, "
                                          "and a response from openweather site."
                                          "Compare the 2 results"
                                          "Provide a message in ukrainian"
                                          "Try to give an advice for what to wear"
                                          "Use emojis and write a lighthearted message"
                                          "Do NOT use Kelvin in you response"},

            {"role": "user", "content": prompt + "\n" + "my result:" + str(out) + "\n"
                            "openweather result:" + str(weather_response)[0:int(len(str(weather_response))/4)]}
        ]
    )

    print(response)
    response_data = response
    return response_data["choices"][0]["message"]["content"].strip()


def get_openai_start_message():
    # Define the API endpoint and headers
    endpoint = "https://api.openai.com/v1/engines/gpt-3.5/completions"
    # Send a request to ChatGPT Turbo 3.5
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a friendly chatbot that tells users a weather forecast."},
            {"role": "user",
             "content": "Please create a friendly message in ukrainian."
                        "Please state that the user needs to provide a location in their request."
                        " Use some emojis."
                        "Dont tell temperature in Kelvin"}
        ]
    )
    # Extract and return the response text
    response_data = response
    return response_data["choices"][0]["message"]["content"].strip()


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when the /start command is received."""
    await update.message.reply_text(get_openai_start_message())


async def handle_responses(update: Update, text: str) -> str:
    """Process the received text and generate a response."""
    return await get_openai_response(update, text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat.id
    text = update.message.text

    response_text = await handle_responses(update, text)
    await update.message.reply_text(response_text)


async def get_weather(update: Update, city: str) -> None:
    url1 = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}"\
           f"&exclude=minutely,daily,alerts&units=metric&appid={WEATHER_API_TOKEN}"
    try:
        response1 = requests.get(url1).json()
        print(response1)

        weather_description =  str(response1)
        return weather_description
    except requests.RequestException:
        return None


async def handle_error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors that occur while processing updates."""
    print(f'Update {update} caused error {context.error.with_traceback()}')


def main():
    logging.basicConfig()  # Improved logging
    logger = logging.getLogger(__name__)

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_error_handler(handle_error)

    logger.info('Starting bot')
    app.run_polling(poll_interval=1)


if __name__ == '__main__':
    main()