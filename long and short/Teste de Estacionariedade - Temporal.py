import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from scipy.stats import zscore
import numpy as np

aD = str(input("Qual o ativo dependente(y)?"))
periodos = ['50d', '75d', '100d', '125d', '150d', '200d', '250d', '300d']
ativos_independentes = ["PETR4.SA", "VALE3.SA", "ABEV3.SA", "AZUL4.SA", "B3SA3.SA", "BBAS3.SA", "BBDC3.SA", "BBDC4.SA", "BBSE3.SA", "BEEF3.SA", "BPAC11.SA", "BRAP4.SA", "BRFS3.SA", "BRKM5.SA", "CCRO3.SA", "CIEL3.SA", "CMIG4.SA", "COGN3.SA", "CPFE3.SA", "CRFB3.SA", "CSAN3.SA", "CSNA3.SA", "CVCB3.SA", "CYRE3.SA", "ECOR3.SA", "EGIE3.SA", "ELET3.SA", "ELET6.SA", "EMBR3.SA", "ENGI11.SA", "EQTL3.SA", "EZTC3.SA", "FLRY3.SA", "GGBR4.SA", "GOAU4.SA", "GOLL4.SA", "HAPV3.SA", "HYPE3.SA", "IRBR3.SA", "ITSA4.SA", "ITUB4.SA", "JBSS3.SA", "KLBN11.SA", "LREN3.SA", "MGLU3.SA", "MRFG3.SA", "MRVE3.SA", "MULT3.SA", "NTCO3.SA", "PCAR3.SA", "PETR3.SA", "PRIO3.SA", "QUAL3.SA", "RADL3.SA", "RAIL3.SA", "RENT3.SA", "SANB11.SA", "SBSP3.SA", "SUZB3.SA", "TAEE11.SA", "TIMS3.SA", "TOTS3.SA", "UGPA3.SA", "USIM5.SA", "WEGE3.SA", "YDUQ3.SA"]  # Lista de ativos independentes

resultados = pd.DataFrame(index=ativos_independentes, columns=periodos)

for per in periodos:
    # IMPORTANDO OS DADOS
    depen = yf.Ticker(aD).history(period=per)["Close"]

    for ativo in ativos_independentes:
        indep = yf.Ticker(ativo).history(period=per)["Close"]

        # Clean the data
        indep = indep.replace([np.inf, -np.inf], np.nan)
        indep = indep.dropna()

        # Perform linear regression
        X = sm.add_constant(indep)  # Independent variable
        y = depen.reindex(indep.index)  # Dependent variable

        # Drop missing values from y
        y = y.dropna()

        if len(X) > 0 and len(y) > 0:
            model = sm.OLS(y, X, missing='drop')
            results = model.fit()

            # Obtenha os resíduos da regressão
            residuos = results.resid

            # Standardize the residuals using z-score
            residuos_padronizados = zscore(residuos)

            central_value0 = 0
            central_value_up0 = 1.8
            central_value_down0 = -1.8

            ultimo_valor_residuais0 = residuos_padronizados.iloc[-1]

            # Clean the standardized residuals
            residuos_padronizados = residuos_padronizados.replace([np.inf, -np.inf], np.nan)
            residuos_padronizados = residuos_padronizados.dropna()

            # Perform the Dickey-Fuller test on the standardized residuals
            if residuos_padronizados.size > 0:
                df_test = sm.tsa.stattools.adfuller(residuos_padronizados)
                p_valor = df_test[1]
                resultados.loc[ativo, per] = p_valor < 0.05
            else:
                print(f"No valid data for Dickey-Fuller test for {ativo} in {per}")
            
            resultados["count"] = resultados.apply(lambda row: row.dropna().astype(bool).sum(), axis=1)-1

            if resultados.loc[ativo, per] and (ultimo_valor_residuais0 > central_value_up0 or ultimo_valor_residuais0 < central_value_down0):
                    resultados.loc[ativo, "entry"] = True
            else: resultados.loc[ativo, "entry"] = False

pd.set_option('display.max_rows', None)

print(resultados)