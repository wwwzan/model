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
df = pd.read_csv(r'./ChinaBank.csv',index_col = 'Date',parse_dates=['Date'])
df = df['2014-01':'2014-04']
# 去掉序号列
df = df.drop(columns=['Unnamed: 0'])
print(df)

# 画出1到4月五个指标的图像
plt.figure(figsize=(15,8))
plt.subplot(321)
plt.title('Open')
plt.xticks(fontsize=7)
plt.plot(df['Open'])

plt.subplot(322)
plt.title('High')
plt.xticks(fontsize=7)
plt.plot(df['High'])

plt.subplot(323)
plt.title('Low')
plt.xticks(fontsize=7)
plt.plot(df['Low'])

plt.subplot(324)
plt.title('Close')
plt.xticks(fontsize=7)
plt.plot(df['Close'])

plt.subplot(313)
plt.title('Volume')
plt.xticks(fontsize=7)
plt.plot(df['Volume'])

plt.tight_layout()
#plt.show()

# step2：因果测试 ， 检验不同序列之间存在互相影响
maxlag=12  #最大延迟阶数
test='ssr_chi2test' #卡方检测
variables=df.columns

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

print(grangers_causation_matrix(df, variables = df.columns))

# step3：ADF测试，检验单个变量是否平稳
def adfuller_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")
for name, column in df.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

# step4: 协整检验，检验多变量平稳性
def adjust(val, length= 6): return str(val).ljust(length)
def cointegration_test(df, alpha=0.05):
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
df1 = df.drop(columns=['Volume'])
#print(df1)
#print(df[:,:4])
print(cointegration_test(df1))

# step5：划分训练集和测试集
df_train, df_test = df1['2014-01':'2014-03'], df1['2014-04':'2014-04']
print(df_train)

# step6：使用VAR之间，先差分处理使单个变量变得平稳
df_differenced = df_train.diff().dropna()
for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

# 差分数据可视化
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(15,8))
for i, ax in enumerate(axes.flatten()):
    data = df_differenced[df_differenced.columns[i]]
    ax.plot(data, linewidth=1)
    ax.set_title(df_differenced.columns[i])
plt.tight_layout()
#plt.show()

# step7：选择模型阶数并训练，根据AIC值，lag=6时达到局部最优
model = VAR(df_differenced)
for i in [1,2,3,4,5,6,7,8,9]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic, '\n')

# 选择lag=6拟合模型
model_fitted = model.fit(6)
model_fitted.summary()
# step8：durbin watson test，检验残差项中是否还存在相关性，这一步的目的是确保模型已经解释了数据中所有的方差和模式
out = durbin_watson(model_fitted.resid)
for col, val in zip(df.columns, out):
    print(adjust(col), ':', round(val, 2))  # 检验值越接近2，说明模型越好

# step9：模型已经足够使用了，下一步进行预测
lag_order = model_fitted.k_ar
forecast_input = df_differenced.values[-lag_order:]
fc = model_fitted.forecast(y=forecast_input, steps=22) #预测22天，表格中4月只有22条数据
df_forecast = pd.DataFrame(fc, index=df1.index[-22:], columns=df1.columns+ '_1d')
print(df_forecast)

# step10：将差分后的值还原为原数据
def invert_transformation(df_train, df_forecast):
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

df_results = invert_transformation(df_train, df_forecast)
# 去掉前面4列
df_results = df_results.drop(columns=['Open_1d','High_1d','Low_1d','Close_1d'])
print("还原结果")
print(df_results)

# 预测差的总和
df_test.columns = ['Open_forecast', 'High_forecast','Low_forecast','Close_forecast']
# print(df_test)
sub = abs(df_results - df_test).sum()
print(sub)

df_results.loc[:, ['Open_forecast', 'High_forecast', 'Low_forecast', 'Close_forecast']]
fig, axes = plt.subplots(nrows=4, ncols=1, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):
    df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    df[col][-22:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()
plt.show()
