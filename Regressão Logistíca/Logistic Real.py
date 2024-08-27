import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta

scaler = MinMaxScaler()
df = pd.DataFrame()

ticker1 = "GOGL34"
ticker2 = "PETR4"
     
MT5Tk1 = ticker1 
MT5Tk2 = ticker2 


acao1 = yf.Ticker(ticker1 + ".SA")
acao2 = yf.Ticker(ticker2 + ".SA")


market = yf.Ticker("BOVA11.SA")

n = 10
time = str(n) + 'y'


df["Stock1"] = acao1.history(period=time, interval='1d')['Close']
df["Stock1Open"] = acao1.history(period=time, interval='1d')['Open']
df["Stock1High"] = acao1.history(period=time, interval='1d')['High']
df["Stock1Low"] = acao1.history(period=time, interval='1d')['Low']
df["Stock2"] = acao2.history(period=time, interval='1d')['Close']
df["Stock2Open"] = acao2.history(period=time, interval='1d')['Open']
df["Stock2High"] = acao2.history(period=time, interval='1d')['High']
df["Stock2Low"] = acao2.history(period=time, interval='1d')['Low']
df["Market"] = market.history(period=time, interval='1d')['Close']

df['maxdist'] = df['Stock1High'].shift(2)/df['Stock1Open']-1
df['mindist'] = df['Stock1Open']/df['Stock1Low'].shift(2)-1

df['maxdist2'] = df['Stock2High'].shift(2)/df['Stock2Open']-1
df['mindist2'] = df['Stock2Open']/df['Stock2Low'].shift(2)-1


df['Sk1Pct'] = df['Stock1'].pct_change()
df['Sk2Pct'] = df['Stock2'].pct_change()
df['MarketPct'] = df['Market'].pct_change()
df['Sk1PctTwo'] = df['Stock1'].pct_change(periods=2)
df['Sk2PctTwo'] = df['Stock2'].pct_change(periods=2)


df['MarketPctTwo'] = df['Market'].pct_change(periods=2)


df['Sk1Acc'] = (1 + df['Sk1Pct']).cumprod()*10000
df['Sk2Acc'] = (1 + df['Sk2Pct']).cumprod()*10000


df['Sk1-MA20'] = df["Stock1"] - df['Stock1'].rolling(window=20).mean()
df['Sk2-MA20'] = df["Stock2"] - df['Stock2'].rolling(window=20).mean()
df["RSI1"] = ta.RSI(df["Stock1"], timeperiod=14)
df["RSI2"] = ta.RSI(df["Stock2"], timeperiod=14)

X = df['Stock1']
y = df['Stock2']
X = sm.add_constant(X)
modelR = sm.OLS(y, X)
resultsR = modelR.fit()
df['residuos'] = resultsR.resid

df['Spread'] = df['Stock1'] - df['Stock2']*resultsR.params[1]
df['Ratio'] = df['Stock1'] / df['Stock2']

df['Ratio-MA10'] = df["Ratio"] - df['Ratio'].rolling(window=10).mean()
df['Ratio-MA20'] = df["Ratio"] - df['Ratio'].rolling(window=20).mean()

df['Alvo' ] = np.where(df['Ratio'].pct_change(periods=2).shift(-2) > 0, 1, 0)

# Média Móvel Convergência Divergência (MACD)
df['MACD'], df['Signal_Line'], _ = ta.MACD(df['Stock1'], fastperiod=12, slowperiod=26, signalperiod=9)

# Diferença para a média móvel de 30 dias
df['Sk1-MA30_diff'] = df['Stock1'] - df['Stock1'].rolling(window=30).mean()
df['Sk2-MA30_diff'] = df['Stock2'] - df['Stock2'].rolling(window=30).mean()

# Desvio padrão dos retornos
df['Sk1_std'] = df['Sk1Pct'].rolling(window=30).std()
df['Sk2_std'] = df['Sk2Pct'].rolling(window=30).std()

# Bandas de Bollinger
df['Sk1_upper_band'], df['Sk1_middle_band'], df['Sk1_lower_band'] = ta.BBANDS(df['Stock1'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
df['Sk2_upper_band'], df['Sk2_middle_band'], df['Sk2_lower_band'] = ta.BBANDS(df['Stock2'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

# Momentum
df['Sk1_momentum'] = df['Stock1'].diff(4)  # 4 dias de diferença
df['Sk2_momentum'] = df['Stock2'].diff(4)

# Taxa de Variação (ROC)
df['Sk1_ROC'] = ta.ROC(df['Stock1'], timeperiod=10)
df['Sk2_ROC'] = ta.ROC(df['Stock2'], timeperiod=10)

# Volatilidade Histórica
df['Sk1_volatility'] = df['Sk1Pct'].rolling(window=30).std() * np.sqrt(252)  # Anualizada
df['Sk2_volatility'] = df['Sk2Pct'].rolling(window=30).std() * np.sqrt(252)  # Anualizada

# Índice de Força Relativa Estocástica (Stochastic RSI)
df['Sk1_stoch_RSI'] = (df['RSI1'] - df['RSI1'].rolling(window=14).min()) / (df['RSI1'].rolling(window=14).max() - df['RSI1'].rolling(window=14).min())
df['Sk2_stoch_RSI'] = (df['RSI2'] - df['RSI2'].rolling(window=14).min()) / (df['RSI2'].rolling(window=14).max() - df['RSI2'].rolling(window=14).min())

# Adicionando a média móvel exponencial (EMA) da coluna 'Ratio'
df['Ratio-EMA10'] = df['Ratio'].ewm(span=10, adjust=False).mean()
df['Ratio-EMA20'] = df['Ratio'].ewm(span=20, adjust=False).mean()

# Adicionando a diferença percentual diária da coluna 'Ratio'
df['Ratio_pct_change'] = df['Ratio'].pct_change()

# Adicionando a volatilidade anualizada da coluna 'Ratio'
df['Ratio_volatility'] = df['Ratio'].rolling(window=30).std() * np.sqrt(252)

# Adicionando o desvio padrão da coluna 'Ratio'
df['Ratio_std'] = df['Ratio'].rolling(window=30).std()

# Adicionando o z-score da coluna 'Ratio'
df['Ratio_z_score'] = (df['Ratio'] - df['Ratio'].mean()) / df['Ratio'].std(ddof=0)

#Volume de Negociação
df['Volume1'] = acao1.history(period=time)['Volume']
df['Volume2'] = acao2.history(period=time)['Volume']

#Diferença Percentual entre Máximo e Mínimo
df['High_Low_Pct_Change1'] = (acao1.history(period=time)['High'] - acao1.history(period=time)['Low']) / acao1.history(period=time)['Low']
df['High_Low_Pct_Change2'] = (acao2.history(period=time)['High'] - acao2.history(period=time)['Low']) / acao2.history(period=time)['Low']

#Correlação
df['Correlation'] = df['Sk1Pct'].rolling(window=30).corr(df['Sk2Pct'])

#Diferença entre EMA e SMA
df['EMA_SMA_Diff1'] = df['Stock1'].ewm(span=10, adjust=False).mean() - df['Stock1'].rolling(window=10).mean()
df['EMA_SMA_Diff2'] = df['Stock2'].ewm(span=10, adjust=False).mean() - df['Stock2'].rolling(window=10).mean()

#Índice de Força (Force Index)
df['Force_Index1'] = df['Stock1'].diff(1) * df['Volume1']
df['Force_Index2'] = df['Stock2'].diff(1) * df['Volume2']

#On-Balance Volume (OBV)
df['OBV1'] = ta.OBV(df['Stock1'], df['Volume1'])
df['OBV2'] = ta.OBV(df['Stock2'], df['Volume2'])

df.dropna(inplace=True)

# Padronize os dados com MinMaxScaler
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

df_scaled.dropna(inplace=True)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE
import numpy as np

df_logic = df_scaled

XL = df_logic.drop('Alvo', axis=1)
yL = df_logic['Alvo']
XL.dropna(inplace=True)
yL.dropna(inplace=True)

def custom_predict(predictions):
    return np.where(predictions > 0.52, 1, np.where(predictions < 0.48, -1, 0))

XL_train, XL_test, yL_train, yL_test = train_test_split(XL, yL, test_size=0.5, random_state=42)
modelL = LogisticRegression()
modelL.fit(XL_train, yL_train)

rfe = RFE(modelL)
fiit = rfe.fit(XL, yL)

cols = fiit.get_support(indices=True)

XL = XL.iloc[:, cols]

XL_train = XL_train.iloc[:, cols]
XL_test = XL_test.iloc[:, cols]

modelL2 = LogisticRegression()
modelL2.fit(XL_train, yL_train)

predictionsL = modelL2.predict(XL_test)
predictionsL = modelL2.predict_proba(XL_test)[:, 1]
predictionsLF = modelL2.predict_proba(XL)[:, 1]
predictionsL = custom_predict(predictionsL)
predictionsLF = custom_predict(predictionsLF)

roc_auc = roc_auc_score(yL, predictionsLF)
rmse = sqrt(mean_squared_error(yL, predictionsLF))


df_resultlog = df[['Stock1','Stock2','Sk1Acc', 'Sk2Acc', 'Sk1PctTwo', 'Sk2PctTwo','Ratio']]
df_resultlog['RatioPct'] = df['Ratio'].pct_change(periods=2)
df_resultlog['PositionLog'] = predictionsLF
df_resultlog['AccLog'] = 5000
df_resultlog['Vol1'] = (df_resultlog['AccLog']//df['Stock1'])
df_resultlog['Vol2'] = (df_resultlog['AccLog']//df['Stock2'])
df_resultlog['SpreadTwo'] = (df_resultlog['Sk1PctTwo']*(df_resultlog['Stock1'].shift(2))*(df_resultlog['Vol1'].shift(2))) - (df_resultlog['Sk2PctTwo']*(df_resultlog['Stock2'].shift(2))*(df_resultlog['Vol2'].shift(2)))
df_resultlog.reset_index(drop=True, inplace=True)

for i in range(2, len(df_resultlog)):
    df_resultlog.loc[i, 'AccLog'] = df_resultlog.loc[i-2, 'AccLog'] + ((df_resultlog.loc[i, 'SpreadTwo'] * df_resultlog.loc[i-2, 'PositionLog']))

df_resultlog['AccLog'] = df_resultlog['AccLog'].rolling(window=2).sum()

df_resultlog['AcuracyCal'] = df_resultlog['PositionLog'].shift(2)

df_resultlog['Acuracy'] = np.where((df_resultlog['AcuracyCal'] * df_resultlog['SpreadTwo']) >= 0, 1, 0)

accuracy = ((df_resultlog['Acuracy'] == 1).sum() / len(df_resultlog))*100

df_resultlog.set_index(df.index, inplace=True)

df_resultlog.dropna(inplace=True)

print(f'Acurácia: {accuracy:.2f}%')

total = ((df_resultlog['AccLog'].iloc[-1] - df_resultlog['AccLog'].iloc[0])/df_resultlog['AccLog'].iloc[0]) * 100
print("O retorno total é de: "f'{total:.2f}'"%")

# anual = ((df_resultlog['AccLog'].iloc[-1]/df_resultlog['AccLog'].iloc[0])**(1/n)-1)*100
anual = total/n
print("O retorno anual é de: "f'{anual:.2f}'"%")

today = date.today()

days_to_subtract = len(XL_test) 

subtract_days = timedelta(days=days_to_subtract)

past_date = today - subtract_days

print(past_date)

import matplotlib.pyplot as plt
import mplcyberpunk

plt.style.use("cyberpunk")

fig, ax = plt.subplots()

ax.plot(df_resultlog.index, df_resultlog['Sk1Acc'], label='Sk1Acc', color='blue')

ax.plot(df_resultlog.index, df_resultlog['Sk2Acc'], label='Sk2Acc', color='red')

ax.plot(df_resultlog.index, df_resultlog['AccLog'], label='AccLog', color='yellow')

ax.axvline(x=past_date, color='green', linestyle='--', label=f"{past_date}")

ax.legend()
plt.show()


print(df_resultlog)