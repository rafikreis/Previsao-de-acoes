import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from scipy.stats import zscore

aD = str(input("Qual o ativo dependente(y)?"))
periodos = ['70d', '100d', '150d', '200d', '250d', '300d']
ativos_independentes = ["PETR4.SA", "VALE3.SA", "ABEV3.SA", "AZUL4.SA", "B3SA3.SA", "BBAS3.SA", "BBDC3.SA", "BBDC4.SA", "BBSE3.SA", "BEEF3.SA", "BPAC11.SA", "BRAP4.SA", "BRFS3.SA", "BRKM5.SA", "CCRO3.SA", "CIEL3.SA", "CMIG4.SA", "COGN3.SA", "CPFE3.SA", "CRFB3.SA", "CSAN3.SA", "CSNA3.SA", "CVCB3.SA", "CYRE3.SA", "ECOR3.SA", "EGIE3.SA", "ELET3.SA", "ELET6.SA", "EMBR3.SA", "ENGI11.SA", "EQTL3.SA", "EZTC3.SA", "FLRY3.SA", "GGBR4.SA", "GOAU4.SA", "GOLL4.SA", "HAPV3.SA", "HYPE3.SA", "IRBR3.SA", "ITSA4.SA", "ITUB4.SA", "JBSS3.SA", "KLBN11.SA", "LREN3.SA", "MGLU3.SA", "MRFG3.SA", "MRVE3.SA", "MULT3.SA", "NTCO3.SA", "PCAR3.SA", "PETR3.SA", "PRIO3.SA", "QUAL3.SA", "RADL3.SA", "RAIL3.SA", "RENT3.SA", "SANB11.SA", "SBSP3.SA", "SUZB3.SA", "TAEE11.SA", "TIMS3.SA", "TOTS3.SA", "UGPA3.SA", "USIM5.SA", "WEGE3.SA", "YDUQ3.SA"]  # Lista de ativos independentes

resultados = pd.DataFrame(index=ativos_independentes, columns=periodos)

for per in periodos:
    # IMPORTANDO OS DADOS
    depen = yf.Ticker(aD).history(period=per)["Close"]

    for ativo in ativos_independentes:
        indep = yf.Ticker(ativo).history(period=per)[
            "Close"].reindex(depen.index)

        # Realize a regressão linear
        X = sm.add_constant(indep)  # Variável independente
        y = depen  # Variável dependente

        model = sm.OLS(y, X, missing='drop')
        results = model.fit()

        # Obtenha os resíduos da regressão
        residuos = results.resid

        # Padronize os resíduos usando z-score
        residuos_padronizados = zscore(residuos)

        # Realize o teste Dickey-Fuller nos resíduos padronizados
        df_test = sm.tsa.stattools.adfuller(residuos_padronizados)

        # O valor-p é o segundo valor retornado pelo teste Dickey-Fuller
        p_valor = df_test[1]

        # Se o valor-p for menor que 0.05, os resíduos são estacionários
        resultados.loc[ativo, per] = p_valor < 0.05

pd.set_option('display.max_rows', None)

print(resultados)
