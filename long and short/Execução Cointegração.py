import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np
from scipy.stats import zscore
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import find_peaks

# ADICIONANDO OS INPUTS
aD = str(input("Qual o ativo dependente(y)?"))
aI = str(input("Qual o ativo independente(x)?"))
per = str(input("Qual o perídodo?"))

# IMPORTANDO OS DADOS
depen = yf.Ticker(aD)
indep = yf.Ticker(aI)

y = depen.history(period=(per))["Close"]
x = indep.history(period=(per))["Close"]

df_calc = pd.DataFrame()
df_calc['y'] = y
df_calc['x'] = x

df_calc['y%'] = (df_calc['y'] - df_calc['y'].shift(1)) * 100
df_calc['x%'] = (df_calc['x'] - df_calc['x'].shift(1)) * 100

x = sm.add_constant(x)

# RODANDO A REGRESSÃO 1
model = sm.OLS(y, x)
results = model.fit()

close_coef = results.params.iloc[1]

# EXTRAINDO OS RESÍDUOS
residuos = results.resid

z = pd.Series(residuos)

# PADRONIZANDO RESÍDUOS PARA O Z SCORE
zscore = zscore(residuos)

# CRIANDO UMA COLUNA DEFAZADA DO Z SCORE PARA RODAR A REGRESSÃO 2
zcdf = pd.Series(zscore)

#CALCULANDO A MEIA VIDA

# halflife = (-np.log(2)) / results.params.iloc[0]

zcdfhf = zcdf

peaks, _ = find_peaks(zcdfhf)

distances = np.diff(peaks)

average_distance = np.mean(distances)

halflife = average_distance

# RODANDO O TESTE DE ESTACIONARIEDADE ADF
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


# print(results.summary())

# print(x, y)

# Limite inferior
limite_inferior = np.std(zcdf)*-1.8

# Limite superior
limite_superior = np.std(zcdf)*1.8

df = pd.DataFrame(zscore, columns=['Z-Score'])

posicao_l = []

# Flag para controle da posição
flag = 1

df_with_original_index = df

df.reset_index(drop=True, inplace=True)

df_calc.reset_index(drop=True, inplace=True)

# Loop para cada linha do DataFrame
for i in range(len(df)):
  # Verifique o valor do resíduo
  if df.loc[i, 'Z-Score'] < limite_inferior:
    posicao_l.append(1)
    flag = 1
  elif df.loc[i, 'Z-Score'] > limite_superior:
    posicao_l.append(-1)
    flag = -1
  else:
    posicao_l.append(flag)

# Adicione a nova coluna "posiçãol" ao DataFrame
df_calc['posiçãol'] = posicao_l

for i in range(len(df)):
    if df_calc.loc[i, 'posiçãol'] == 1:
        df_calc.loc[i, 'Long'] = (+(df_calc.loc[i, 'y%']*close_coef)-(df_calc.loc[i, 'x%']))
        df_calc.loc[i, 'Short'] = 0

    elif df_calc.loc[i, 'posiçãol'] == -1:
        df_calc.loc[i, 'Short'] = (-(df_calc.loc[i, 'y%']*close_coef)+(df_calc.loc[i, 'x%']))
        df_calc.loc[i, 'Long'] = 0

df_calc['Acc'] = 0

for i in range(1, len(df)):
   df_calc.loc[i,'Acc'] = df_calc.loc[i, 'Long'] + df_calc.loc[i, 'Short'] + df_calc.loc[i - 1, 'Acc']

print(df_calc)

# Plot the 'Acc' column
df_calc['Acc'].plot()

# Add labels and title

df.plot(kind='line', color='blue')
plt.axhline(y=limite_superior, color='r', linestyle='--', label='2 Desvios Padrão Acima')
plt.axhline(y=limite_inferior, color='g', linestyle='--', label='2 Desvios Padrão Abaixo')

plt.xlabel('Index')
plt.ylabel('Acc')
plt.title('Acc Over Time')

print(test_stationarity(zcdf))

# print(resz.summary())

print('Meia-vida = {:.2f}'.format(halflife))

print("A proporção do ativo X é {:.2f} pra 1 de Y".format(close_coef))

# Show the plot
plt.show()