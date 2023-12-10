import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np

# ADICIONANDO OS INPUTS
aD = str(input("Qual o ativo dependente(y)?"))
aI = str(input("Qual o ativo independente(x)?"))
per = str(input("Qual o perídodo?"))

# IMPORTANDO OS DADOS
depen = yf.Ticker(aD)
indep = yf.Ticker(aI)

y = depen.history(period=(per))["Close"]
x = indep.history(period=(per))["Close"]
x = sm.add_constant(x)

# RODANDO A REGRESSÃO 1
model = sm.OLS(y, x)
results = model.fit()

close_coef = results.params.iloc[1] * -1

y_pred = results.predict(x)

# EXTRAINDO OS RESÍDUOS
residuos = y - y_pred

z = pd.Series(residuos)

# PADRONIZANDO RESÍDUOS PARA O Z SCORE
media = z.mean()
desvpad = z.std()

zscore = (z - media) / desvpad

# CRIANDO UMA COLUNA DEFAZADA DO Z SCORE PARA RODAR A REGRESSÃO 2
zcdf = pd.Series(zscore)

zcdf_lag = np.roll(zcdf, 1)
zcdf_lag[0] = 0

# RODANDO REGRESSÃO 2
modelz = sm.OLS(zcdf_lag, zcdf)
resz = modelz.fit()

# CALCULANDO A MEIA VIDA
halflife = -np.log(2) / resz.params.iloc[0]

# RODANDO O TESTE DE ESTACIONARIEDADE
adf = adfuller(zcdf)


def test_stationarity(zcdf, significance_level=0.05):
    print('ADF Statistic: %f' % adf[0])
    print('p-value: %f' % adf[1])
    print('Critical Values:')
    for key, value in adf[4].items():
        print('\t%s: %.3f' % (key, value))
    if adf[1] < significance_level:
        print("A série é estacionária")
    else:
        print("A série é não-estacionária")


print(test_stationarity(zcdf))

# print(resz.summary())

print('Meia-vida = {:.2f}'.format(halflife))

print("A proporção do ativo X é {:.2f} pra 1 de Y".format(close_coef))

# print(results.summary())

# print(x, y)

df = pd.DataFrame(zscore, columns=['Z-Score'])

df.plot(kind='line', color='blue')

plt.show()
