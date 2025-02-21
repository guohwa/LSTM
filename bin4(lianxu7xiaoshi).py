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
    df = df.drop(["timestamp", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"], axis=1)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric)

    # 保存数据到CSV文件
    df.to_csv("binance_data.csv", index=False)

    # 数据预处理
    df = pd.read_csv("binance_data.csv")  # 读取CSV文件到DataFrame
    df = df.dropna()  # 删除包含缺失值的行
    df["close"] = df["close"].pct_change()  # 计算“close”列的百分比变化
    df = df.dropna()

    return df

# ...existing code...
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
    for i in range(60, len(train_data) - 6):  # 修改这里，确保有足够的数据用于预测7小时
        X_train.append(train_data[i-60:i])
        y_train.append(train_data[i:i+7, 0])  # 修改这里，预测接下来7小时的价格
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = [], []
    for i in range(60, len(test_data) - 6):  # 修改这里，确保有足够的数据用于预测7小时
        X_test.append(test_data[i-60:i])
        y_test.append(test_data[i:i+7, 0])  # 修改这里，预测接下来7小时的价格
    X_test, y_test = np.array(X_test), np.array(y_test)

    # 创建LSTM模型
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(7))  # 修改这里，输出7个值
    model.compile(loss="mae", optimizer="adam")

    # 训练模型
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1, shuffle=False)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    y_pred = y_pred * train_std[0] + train_mean[0]
    y_test = y_test * train_std[0] + train_mean[0]

    # 计算每个时间点的平均值
    y_test_mean = y_test.mean(axis=1)
    y_pred_mean = y_pred.mean(axis=1)

    # 绘制测试集上的实际值和预测值图
    plt.figure(figsize=(16,8))
    plt.plot(y_test_mean, label="actual")
    plt.plot(y_pred_mean, label="predicted")

    # 获取测试集的最后60个值，转换为3D数组并输入到训练好的LSTM模型中，预测下一个值
    last_60 = test_data[-60:]

    # 将最后60个值转换为3D数组
    last_60 = np.array(last_60)
    last_60 = np.reshape(last_60, (1, last_60.shape[0], last_60.shape[1]))

    # 预测下一个值
    next_values = model.predict(last_60)

    # 反标准化预测值
    next_values = next_values * train_std[0] + train_mean[0]

    # 在图表上标记接下来7个小时的预测值
    for i in range(7):
        plt.plot(len(y_test_mean) + i, next_values[0][i], marker="o", markersize=10, label=f"next value {i+1}h" if i == 0 else "")

    plt.legend()
    plt.show()

    # 打印预测值
    for i in range(7):
        print(f"接下来第{i+1}小时价格预测: {next_values[0][i]}")

    # 保存模型
    model.save("model.keras")

# 循环训练和预测
while True:
    print(f"---------------------------》》-------分隔线----------》》---------------- ")
    print(f"开始新的训练和预测循环: {datetime.now()}")
    train_and_predict()
    print(f"训练和预测循环结束，训练结束时间: {datetime.now()}")
    time.sleep(3600)  # 每隔1小时训练一次