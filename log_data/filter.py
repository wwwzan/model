import pandas as pd

# 读取Excel文件
df = pd.read_excel(r'D:\桌面\data\2-3\2-3.xlsx')

# 设置起始时间
start_time = pd.to_datetime('2024-5-1')

# 计算行数
num_entries = len(df)

# 生成时间列，每0.4秒一个时间点
time_series = pd.date_range(start=start_time, periods=num_entries, freq='10s')

# 将时间列添加到DataFrame
df.insert(0, 'Time', time_series)

# 对流量、密度和速度数据进行平滑处理（使用滚动窗口方法）
# 使用滚动窗口（window=5）对流量、密度、速度进行平滑，调整窗口大小以适应你的数据
df['Flow_Smoothed'] = df['流量'].rolling(window=5).mean()
df['Density_Smoothed'] = df['密度'].rolling(window=5).mean()
df['Speed_Smoothed'] = df['平均速度'].rolling(window=5).mean()

# 去除速度超过150的行
# filtered_df = df[df['Speed (km/h)'] <= 150]

# # 根据60秒输出一次结果，选择合适的行
# # 每60秒对应150行（60秒 / 0.4秒 = 150行）
# output_df = df.iloc[::25]

# 将最终结果写入新的Excel文件
df.to_excel('filtered_vehicle_data_with_time2_3_smoothed.xlsx', index=False)

print("极端值已成功去除，并保存为filtered_vehicle_data_with_time1.xlsx")
