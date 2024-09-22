import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
import warnings
import datetime

warnings.filterwarnings("ignore")
# step1：导入数据，数据可视化
df = pd.read_csv('./ChinaBank.csv',index_col = 'Date',parse_dates=['Date'])
df = df['2014-01':'2014-04']
# 去掉序号列和前面4列
df = df.drop(columns=['Unnamed: 0','Open','High','Low','Close'])
# print(df)

df_train = df.loc['2014-01':'2014-03']
df_test = df.loc['2014-04':'2014-04']
print(df_train)

# 选择ARIMA模型，用ACF和PACF判断参数（0阶）
import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_train, lags=20,ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout()
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_train, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout()
# plt.show()

#遍历，寻找适宜的参数
import itertools
import numpy as np
import seaborn as sns
p_min = 0
d_min = 0
q_min = 0
p_max = 5
d_max = 0
q_max = 5
# Initialize a DataFrame to store the results,，以BIC准则
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
 columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])
for p,d,q in itertools.product(range(p_min,p_max+1),
 range(d_min,d_max+1),
 range(q_min,q_max+1)):
 if p==0 and d==0 and q==0:
     results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
     continue
 try:
     #模型
     model = sm.tsa.ARIMA(df_train, order=(p, d, q),
     #enforce_stationarity=False,
     #enforce_invertibility=False,
     )
     results = model.fit()
     results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
 except:
        continue
results_bic = results_bic[results_bic.columns].astype(float)
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(results_bic,
mask=results_bic.isnull(),
ax=ax,
annot=True,
fmt='.2f',
 )
ax.set_title('BIC')
# plt.show()

 # MA0和AR1
 #重构数据，将index设计成数而不是时间
new_df = pd.read_csv('./ChinaBank.csv')
# 去掉序号列和前面4列
new_df = new_df.drop(columns=['Unnamed: 0','Date','Open','High','Low','Close'])
train = new_df.loc[0:61]
print(train)
test = new_df.loc[62:83]

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
fig, ax = plt.subplots(figsize=(12, 8))
# 用训练集
ax = train.loc[0:].plot(ax=ax)
res = ARIMA(train,order=(1,0,0)).fit()
fig = plot_predict(res,34, 83, dynamic=False, ax=ax, plot_insample=False)
plt.show()
print(res.summary())

# 预测结果
predict=res.predict(62, 83)
print(predict)

# 预测差的总和
# 将预测结果变成DataFrame
sub = abs(predict - test['Volume']).sum()
print("Volume    "+str(sub))