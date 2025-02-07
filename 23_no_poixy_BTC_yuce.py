import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import requests
import sqlite3
import os
import time
import logging
from datetime import datetime, timedelta

# 设置日志记录
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 获取 Binance K线数据
def fetch_data():
    try:
        response = requests.get("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1000", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"请求失败: {e}")
        return None

# 将数据存储到SQLite数据库
def store_data_to_db(data):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    # 删除旧的表（如果存在）
    cursor.execute("DROP TABLE IF EXISTS predictions")
    
    # 创建新的表
    cursor.execute('''CREATE TABLE predictions
                      (date TEXT, predicted_price REAL)''')
    
    for entry in data:
        date = entry[0]  # 保持原始时间戳格式
        predicted_price = float(entry[1])  # 确保为浮点数
        logging.info(f"Inserting into DB: date={date}, predicted_price={predicted_price}")  # 调试信息
        cursor.execute("INSERT INTO predictions (date, predicted_price) VALUES (?, ?)", (date, predicted_price))
    
    conn.commit()
    conn.close()

# 主训练和预测流程
def train_and_predict():
    logging.info("开始训练和预测流程")
    
    # 获取数据
    data = fetch_data()
    if data is None:
        logging.error("未能获取数据，程序退出。")
        return

    # 转换为 DataFrame
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume",
                                     "close_time", "quote_asset_volume", "number_of_trades",
                                     "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric)

    # 划分训练和测试数据
    train_size = int(len(df) * 0.8)
    train_data = df[["open", "high", "low", "close", "volume"]][:train_size].values
    test_data = df[["open", "high", "low", "close", "volume"]][train_size:].values

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # 构造时间序列数据
    def create_dataset(data, time_step=60):
        X, Y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), :])  # 使用多个特征
            Y.append(data[i + time_step, 3])  # 预测收盘价
        return np.array(X), np.array(Y)

    time_step = 60
    X_train, y_train = create_dataset(train_scaled, time_step)
    X_test, y_test = create_dataset(test_scaled, time_step)

    # 调整形状
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    # 构建 LSTM 模型
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    # 训练模型
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    # 预测
    predicted_prices = model.predict(X_test)
    predicted_prices_reshaped = predicted_prices.reshape(-1, 1)
    predicted_prices_final = scaler.inverse_transform(np.hstack((test_scaled[time_step:, :4], predicted_prices_reshaped)))

    # 预测下一小时价格
    last_60_hours = test_scaled[-time_step:].reshape(1, time_step, 5)
    predicted_next_hour_price = model.predict(last_60_hours)
    predicted_next_hour_price_reshaped = predicted_next_hour_price.reshape(-1, 1)

    # 计算当前时间的下一小时
    current_time = datetime.now()
    next_hour_date = current_time + timedelta(hours=1)

    # 确保预测结果有效
    predicted_next_hour_price_final = None
    if predicted_next_hour_price_reshaped is not None:
        predicted_next_hour_price_final = scaler.inverse_transform(np.hstack((last_60_hours[0, -1, :4].reshape(1, -1), predicted_next_hour_price_reshaped)))

    # 将预测结果存储到数据库
    if predicted_next_hour_price_final is not None:
        store_data_to_db([(next_hour_date.strftime('%Y-%m-%d %H:%M:%S'), predicted_next_hour_price_final[0][0])])

    # 打印预测结果
    if predicted_next_hour_price_final is not None:
        print(f"预测日期和时间: {next_hour_date}, 预测价格: {predicted_next_hour_price_final[0][0]}")
        logging.info(f"预测日期和时间: {next_hour_date}, 预测价格: {predicted_next_hour_price_final[0][0]}")

    # 可视化
    plt.figure(figsize=(14, 5))
    plt.plot(df.index[:train_size + time_step], df["close"][:train_size + time_step], label="Historical Prices")
    plt.plot(df.index[train_size + time_step:], predicted_prices_final[:, 0], label="Predicted Prices", color='red')
    plt.scatter(df.index[train_size:], df["close"][train_size:], label="Test Set", color='blue')
    if predicted_next_hour_price_final is not None:
        plt.scatter(next_hour_date, predicted_next_hour_price_final[0][0], color='red', s=30, label="Predicted Next Hour Price")
    plt.title("Bitcoin Price Prediction (Based on Binance Data)", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price (USD)", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

    logging.info("训练和预测流程结束")

# 循环训练
while True:
    logging.info("开始新的训练循环")
    train_and_predict()
    logging.info("训练循环结束，等待1小时")
    time.sleep(3600)  # 每隔1小时训练一次
