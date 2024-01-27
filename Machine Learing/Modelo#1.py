import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd

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

a1['mercado'] = ibov

a1 = a1.dropna()

a2 = a1.drop(['retornos', 'pos', 'neg', 'pos_med', 'neg_med'], axis=1)

y1 = a2.drop(['Mean', 'ifr7', 'mercado'], axis=1)
y2 = a1['retornos'].apply(lambda x: 1 if x > 0 else 0).shift(periods=1)

meany2 = y2.mean()

y2 = y2.fillna(meany2)

X_train, X_test, y_train, y_test = train_test_split(a2, y2, test_size=0.2)

lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y_train)

model = LogisticRegression()
model.fit(X_train, y_transformed)

y_pred = model.predict(X_test)

print(y_pred)

acuracia = accuracy_score(y_test, y_pred)

dados = {'Retornos': a1.loc[X_test.index, 'retornos'],
         'Predição': y_pred
         }

df = pd.DataFrame(dados)

# Exiba o DataFrame
print(df)

print(f"A taxa de acerto do modelo é: {acuracia}")
