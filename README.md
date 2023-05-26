# Stock-Predictor-V4

![stockpredictor ai logo](https://user-images.githubusercontent.com/53996451/224323224-3ec1cd20-747c-42ad-9fb1-ba6e0ecb358b.png)

---
# Content Table

- [Stock-Predictor-V4](#stock-predictor-v4)
  - [1a. Installation](#1a-installation)
  - [1b. Installation with Virtual Environment (venv) in Python](#1b-installation-with-virtual-environment-venv-in-python)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Training the LSTM RL Model](#3-training-the-lstm-rl-model)
  - [4. Evaluating the Model](#4-evaluating-the-model)
  - [5. Fine Tuning the LSTM RL Model](#5-fine-tuning-the-lstm-rl-model)
  - [6. Utilizing the Model for Stock Market Prediction](#6-utilizing-the-model-for-stock-market-prediction)
  - [7. Comparing the predicted values with the actual values after the 30-day period.](#7-comparing-the-predicted-values-with-the-actual-values-after-the-30-day-period)

---

## 1a. Installation
```
git clone https://github.com/Qerim-iseni09/Stock-Predictor-V4.git
cd Stock-Predictor-V4
chmod +x install.sh
./install.sh # Ignore the 2 Messages
```

---

## 1b. Installation with Virtual Environment (venv) in Python
```
git clone https://github.com/Qerim-iseni09/Stock-Predictor-V4.git
pip install venv
python -m venv Stock-Predictor-V4
cd Stock-Predictor-V4
source bin/activate # Execute before using any script
./install.sh # Ignore the 2 Messages
deactivate # Execute this command to deactivate the virtual environment
```

---

## 2. Data Preparation
To prepare the data for your stock prediction, follow these steps:

Option 1:
1. Go to [Yahoo Finance](https://finance.yahoo.com/) and select a stock of your choice.
2. Download the CSV file from the Historical Data tab and save it in the data folder. For example, you can use [Bitcoin](https://finance.yahoo.com/quote/BTC-USD?p=BTC-USD) as an example.
3. Open `prepare_data.py` and change `df = pd.read_csv('data/example.csv')` to `df = pd.read_csv('data/<Your Downloaded CSV file>')`, where `<Your Downloaded CSV file>` is the name of the CSV file you downloaded in step 2.
4. Execute `prepare_data.py` by running `python prepare_data.py`.

Option 2:
1. Use `gen_stock.py` to generate Fake Stock Data to train your model for any option you chose.
2. Execute `prepare_data.py` by running `python prepare_data.py`.

After these steps, your downloaded/generated stock data will have indicators that make predictions more reliable.

---

## 3. Training the LSTM RL Model

To train the LSTM RL Model with the `data.csv` file that was generated by the 2. Step, execute the following:

```
python train.py
```
After running this command, the LSTM RL model will start training with the data in the data.csv file.

---

## 4. Evaluating the Model
To evaluate the trained model, execute the following:

```
python eval.py
```

After running this command, the root mean squared error (RMSE), the MAPE and Total Rewards will be plotted.

---

## 5. Fine Tuning the LSTM RL Model

To fine tune the model, execute the following:
```
python fine_tune.py
```
Before starting the fine-tuning process for the LSTM RL model, the script will now ask you to specify your desired reward threshold. It is recommended to set the threshold at 0.9.

While waiting for the fine-tuning process to finish, it is advisable to stay hydrated by drinking some water, as this process may take some time. Once the fine-tuning is complete, try running the "fine_tune.py" script until it loops only once. In case it doesn't loop once initially, continue rerunning it until it completes at least one loop.

After the fine-tuning process is fully complete, it is highly recommended to re-evaluate the model's performance.

---

## 6. Utilizing the Model for Stock Market Prediction
Once the previous steps have been completed, the model can be utilized to forecast the stock market for the next 30 days beyond the latest date in the data. The predictions will be shown in the command line and saved as a CSV file.

To use the model for prediction, run the following command:

```
python predict.py
```

And Happy Trading!
However, please note that any losses incurred by utilizing the model's predictions are not the responsibility of the developer.

---

## 7. Comparing the predicted values with the actual values after the 30-day period.

If you've reached the end of the 30-day predicted period and you're curious about the accuracy of the model, then the `compare.py` script can help you. To compare the data, you need to update the CSV file you selected in Step 2. Open the file and change `actual_data = pd.read_csv(os.path.join("data", "BTC-USD.csv"))` to `actual_data = pd.read_csv(os.path.join("data", "<your downloaded CSV file>"))`.

After that, run the following command in your terminal:
```
python compare.py
```

This will compare the predicted data with the actual data. You may regret some of the decisions you made and wonder why you didn't trust the model (just kidding, don't take it too seriously!).
