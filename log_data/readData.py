import pandas as pd
import re

# 读取文件
with open('4_2.txt', 'r') as file:
    lines = file.readlines()

# 存储结果的列表
results = []

# 遍历每一行
for line in lines:
    # 使用正则表达式提取速度和车辆数量
    speed_match = re.search(r'(\d+\.\d+)ms', line)
    vehicle_match = re.findall(r'(\d+)\s+(cars?|buses?|trucks?)', line)

    if speed_match and vehicle_match:
        speed = float(speed_match.group(1))
        total_vehicles = sum(int(v[0]) for v in vehicle_match)

        results.append({
            'Speed (km/h)': speed,
            'Total Vehicles': total_vehicles
        })

# 创建DataFrame并写入Excel
df = pd.DataFrame(results)
print(df)
df.to_excel('vehicle4_2data.xlsx', index=False)

print("数据已写入Excel文件。")



