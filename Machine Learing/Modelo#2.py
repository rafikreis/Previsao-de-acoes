import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

acao = 'petr4.sa'

a = yf.Ticker(acao)

a1 = a.history(period="10y", interval='1d')

a1 = a1.drop(['High', 'Low', 'Open', 'Dividends',
             'Stock Splits', 'Volume'], axis=1)

a1['Mean'] = a1.rolling(window=50).mean()

a1['retornos'] = a1['Close'].pct_change().dropna()

a1['pos'] = a1['retornos'].apply(lambda x: x if x > 0 else 0)
a1['neg'] = a1['retornos'].apply(lambda x: abs(x) if x < 0 else 0)

a1['pos_med'] = a1['pos'].rolling(2).mean()
a1['neg_med'] = a1['neg'].rolling(2).mean()

a1['ifr7'] = 100 - 100/(1 + a1['pos_med']/a1['neg_med'])

ibov = yf.Ticker('^BVSP').history(period="10y", interval='1d')

ibov.drop(['High', 'Low', 'Open', 'Dividends',
          'Stock Splits', 'Volume'], axis=1)

ibov = ibov.drop(['High', 'Low', 'Open', 'Dividends',
                 'Stock Splits', 'Volume'], axis=1)

a1['Vol'] = a1['retornos'].rolling(7).std()

a1['mercado'] = ibov

a1['MeanRet'] = a1['Mean'].pct_change().dropna()
a1['MercadoRet'] = a1['mercado'].pct_change().dropna()

a1 = a1.dropna()

a2 = a1.drop(['pos', 'neg', 'pos_med', 'neg_med'], axis=1)

meana2 = a2['retornos'].mean()

y = a2['retornos'].shift(periods=-1)
y = y.fillna(meana2)
X = a2.drop(['retornos', 'Close', 'Mean', 'mercado'], axis=1)

seed = 42
np.random.seed(seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print(f'O erro quadrático médio das previsões é: {mse}')

y_test_df = pd.DataFrame(y_test)
y_test_df['direcao_real'] = (y_test_df['retornos'] > 0).astype(int)

y_pred_df = pd.DataFrame(y_pred, index=y_test_df.index, columns=['retornos'])
y_pred_df['direcao_predita'] = (y_pred_df['retornos'] > 0).astype(int)

resultado = pd.concat([y_test_df, y_pred_df], axis=1)
resultado.columns = ['retornos_real', 'direcao_real', 'retornos_predito', 'direcao_predita']
resultado['acerto'] = (resultado['direcao_real'] ==
                       resultado['direcao_predita']).astype(int)

taxa_de_acerto = resultado['acerto'].mean()

print(f'A taxa de acerto do modelo é: {taxa_de_acerto}')

resultado['retorno_acumulado'] = np.where(resultado['retornos_real'] > 0, resultado['retornos_real'], -resultado['retornos_real']).cumsum()

print(resultado)

plt.figure(figsize=(10, 6))
plt.plot(resultado.index, resultado['retorno_acumulado'], label='Model Accumulated Return', color='blue')
plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Zero Line')
plt.title('Accumulated Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Accumulated Return')
plt.legend()
plt.grid(True)
plt.show()
