# 导入必要的库
import requests
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from datetime import datetime, timedelta

# 代理配置
proxies = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890"
}

# Binance API设置
url = "https://api.binance.com/api/v3/klines"
symbol = "BTCUSDT"
interval = "1h"
limit = "1000"

def fetch_data():
    # 获取数据
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params, proxies=proxies)
    data = response.json()

    # 创建DataFrame并指定列名
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df.set_index("timestamp", inplace=True)
    df = df.drop(["close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"], axis=1)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric)

    return df

def train_and_predict():
    df = fetch_data()

    # 将原始数据集按80/20的比例划分为训练集和测试集
    split = int(0.8 * len(df))
    train_df = df[:split]
    test_df = df[split:]

    # 对数据进行标准化处理，以训练集的均值和标准差为基准
    train_mean = train_df.mean()
    train_std = train_df.std()
    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    # 转换为3D数组，每个元素表示前60个值的序列，用于训练模型
    train_data = train_df.values
    test_data = test_df.values
    X_train, y_train = [], []
    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i])
        y_train.append(train_data[i, 3])  # 使用收盘价作为目标值
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = [], []
    for i in range(60, len(test_data)):
        X_test.append(test_data[i-60:i])
        y_test.append(test_data[i, 3])  # 使用收盘价作为目标值
    X_test, y_test = np.array(X_test), np.array(y_test)

    # 创建LSTM模型
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss="mae", optimizer="adam")

    # 训练模型
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1, shuffle=False)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    y_pred = y_pred * train_std[3] + train_mean[3]  # 反标准化预测值
    y_test = y_test * train_std[3] + train_mean[3]  # 反标准化实际值

    # 绘制测试集上的实际值和预测值图
    plt.figure(figsize=(16,8))
    plt.plot(df.index[-len(y_test):], y_test, label="actual")
    plt.plot(df.index[-len(y_pred):], y_pred, label="predicted")
    plt.legend()
    plt.show()

    # 评估模型在测试集上的性能
    mse = mean_squared_error(y_test, y_pred)  # 计算均方误差
    mae = mean_absolute_error(y_test, y_pred)  # 计算平均绝对误差
    print("测试集上的MSE(反标准化):", mse)
    print("测试集上的MAE(反标准化):", mae)

    # 获取测试集的最后60个值，转换为3D数组并输入到训练好的LSTM模型中，预测下一个值
    last_60 = test_data[-60:]

    # 将最后60个值转换为3D数组
    last_60 = np.array(last_60)
    last_60 = np.reshape(last_60, (1, last_60.shape[0], last_60.shape[1]))

    # 预测下一个值
    next_value = model.predict(last_60)

    # 反标准化预测值
    next_value = next_value * train_std[3] + train_mean[3]

    # 打印预测值
    print(f"预测的下一个值: {next_value[0][0]}")

    # 显示预测值的图表
    plt.figure(figsize=(16,8))
    plt.plot(df.index[-len(y_test):], y_test, label="actual")
    plt.plot(df.index[-len(y_pred):], y_pred, label="predicted")
    plt.plot(df.index[-1] + pd.Timedelta(hours=1), next_value, marker="o", markersize=10, color="green", label="next value")
    plt.legend()
    plt.show()

    # 保存模型
    model.save("model.keras")

# 循环训练和预测
while True:
    print(f"---------------------------》》-------分隔线----------》》---------------- ")
    print(f"开始新的训练和预测循环: {datetime.now()}")
    train_and_predict()
    print(f"训练和预测循环结束，训练结束时间: {datetime.now()}")

    # 计算下次训练时间
    now = datetime.now()
    next_train_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    sleep_time = (next_train_time - now).total_seconds()
    print(f"下次训练时间: {next_train_time}")
    time.sleep(sleep_time)  # 等待到下次训练时间