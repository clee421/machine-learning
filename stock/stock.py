import os
from pathlib import Path

import torch
from torch import nn

from typing import List, TypedDict

import pandas as pd
import numpy as np

import yfinance as yf

SPOTIFY_STOCK = "SPOT"
SPOTIFY_DATA_FILEPATH = os.path.join("data", "spot.csv")


class Data(TypedDict):
    inputs: torch.Tensor
    label: torch.Tensor

def write(data, filepath):
  # Ensuring the filepath exists
  Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)

  file_object = open(filepath, "w")
  file_object.write(data)
  file_object.close()

def fetch_daily_stock_data(ticker_symbol):
  # https://github.com/ranaroussi/yfinance
  ticker = yf.Ticker(ticker_symbol)

  # all data available for ticker
  ticker_history = ticker.history(period="max")

  return ticker_history

def fetch_spotify_stock_data(overwrite = False):
    if not os.path.exists(SPOTIFY_DATA_FILEPATH) or overwrite:
        print("Data not found, fetching from yahoo finance...")
        dataframe = fetch_daily_stock_data(SPOTIFY_STOCK)
        print(f"Writing data to ${SPOTIFY_DATA_FILEPATH}")
        write(dataframe.to_csv(), SPOTIFY_DATA_FILEPATH)
    else:
        print(f"Reading data from ${SPOTIFY_DATA_FILEPATH}")
        dataframe = pd.read_csv(SPOTIFY_DATA_FILEPATH)

    return dataframe

def create_training_set_from_stock_data(spotify_df):
    spotify_df["Tomorrow"] = spotify_df["Close"].shift(-1)
    spotify_df["Target"] = (spotify_df["Tomorrow"] > spotify_df["Close"]).astype(int)

    # The dividens and stock splits aren't that useful
    del spotify_df["Dividends"]
    del spotify_df["Stock Splits"]

    # We're going to drop the date for now unless it's meaningful to convert
    # to a tensor
    del spotify_df["Date"]

    # We don't have access to tomorrow's price so we remove it once we know the target
    del spotify_df["Tomorrow"]

    # Drop last row because tomorrow / target is not a valid result
    spotify_df = spotify_df.iloc[:-1]

    spotify_np = spotify_df.to_numpy()

    training_data = []
    testing_data = []
    batch_size = 1
    # creating batches
    for i in range(0, len(spotify_np), batch_size):
        batch = spotify_np[i:i+batch_size].flatten()

        # The last item is target / label
        label = batch[-1]
        batch = np.delete(batch, -1)
        # We're going with a 2:1 ratio of training to validation
        if (i+1) % 3 == 0:
            testing_data.append({
                "inputs": torch.tensor(batch),
                "label": torch.tensor(label),
            })
        else:
            training_data.append({
                "inputs": torch.tensor(batch),
                "label": torch.tensor(label),
            })

    return training_data, testing_data

# TODO: Update to LSTM model
def build_model(
    num_features: int,
    hidden_layers: List[int] = [100, 75, 50, 25]
):
    layers = []
    input_features = num_features
    for output_features in hidden_layers:
        layers.append(nn.Linear(input_features, output_features, dtype=torch.float64))
        # https://universepg.com/public/storage/journal-pdf/Determining%20the%20best%20activation%20functions%20for%20predicting%20stock%20prices.pdf
        # TanH seems to be yield good results for activation function
        layers.append(nn.Tanh())
        input_features = output_features

    layers.append(nn.Linear(input_features, 1, dtype=torch.float64))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)

def training_function(ds: List[Data], model: nn.Module, epochs: int):
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


    for epoch in range(epochs):

        running_loss = 0.0
        for index, data in enumerate(ds):
            inputs = data["inputs"]
            label = data["label"].unsqueeze(-1)

            optimizer.zero_grad()

            y_pred = model(inputs)
            loss = loss_fn(y_pred, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if index % 100 == 99:
                print(f"[{epoch + 1}, {index + 1:4d}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

def evaluate_function(ds: List[Data], model: nn.Module):
    model.eval()
    loss_fn = torch.nn.BCELoss()

    running_loss = 0.0
    for index, data in enumerate(ds):
        inputs = data["inputs"]
        label = data["label"].unsqueeze(-1)


        y_pred = model(inputs)
        loss = loss_fn(y_pred, label)

        loss.backward()

        running_loss += loss.item()
        if index % 100 == 99:
            print(f"[{index + 1:4d}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

    model.train()

def main():
    spotify_df = fetch_spotify_stock_data()
    training_data, validation_data = create_training_set_from_stock_data(spotify_df)

    print(f"Total Size: {len(training_data) + len(validation_data)}")
    print(f"Train Size: {len(training_data)} / Test Size {len(validation_data)}")

    model = build_model(list(training_data[0]["inputs"].size())[-1])

    epochs = 5
    training_function(training_data, model, epochs)

    evaluate_function(validation_data, model)


if __name__ == "__main__":
  main()