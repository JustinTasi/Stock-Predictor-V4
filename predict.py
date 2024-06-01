import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
import tensorflow as tf

# 加载模型
model = tf.keras.models.load_model('model.keras')

# 加载和准备数据
data = pd.read_csv('data.csv')
close_prices = data['Close'].values.reshape(-1, 1)

# 归一化
scaler = MaxAbsScaler()
scaled_data = scaler.fit_transform(close_prices)

# 选择最后的时间窗口数据作为输入
timesteps = 100
last_sequence = scaled_data[-timesteps:]
last_sequence = np.expand_dims(last_sequence, axis=0)

# 预测未来N天
future_days = 30
predictions = []
for _ in range(future_days):
    # 获取最新预测
    current_pred = model.predict(last_sequence)
    
    # 调整 current_pred 形状以匹配 last_sequence
    current_pred_reshaped = np.expand_dims(current_pred, axis=-1)  # 使其成为 (1, 1, 1)
    
    # 更新输入数据
    last_sequence = np.append(last_sequence[:, 1:, :], current_pred_reshaped, axis=1)

    # 保存预测结果以反归一化
    predictions.append(current_pred_reshaped.flatten()[0])

# 反归一化预测结果
predictions = np.array(predictions).reshape(-1, 1)
predicted_prices = scaler.inverse_transform(predictions)
predicted_prices = predicted_prices.flatten()  # 展平数组

# 打印预测结果
print("预测的未来 {} 天的价格是:".format(future_days))
print(predicted_prices)
