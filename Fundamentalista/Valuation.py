import fundamentus as fd
import pandas as pd

ticker = fd.get_resultado()

x = int(input("Escolha qual tipo de ação você deseja: \n 1-Dividendos \n 2-Alto lucro \n 3-Baratas \n:"))

if x == 1:
    a1 = ticker.drop(['roic', 'pvp', 'pcg', 'mrgliq', 'pebit', 'pacl',
                     'evebit', 'evebitda', 'mrgebit', 'liq2m', 'psr', 'pa', 'patrliq'], axis=1)
    filtro = (a1['dy'] > 0.05) & (a1['divbpatr'] < 2) & (
        a1['c5y'] > 0.05) & (a1['liqc'] > 1) & (a1['roe'] > 0.10)
    a2 = a1[filtro]

    maxmin = (a2 - a2.min()) / (a2.max() - a2.min())
    maxmin['divbpatr'] = (maxmin['divbpatr'] - 1) * -1
    maxmin['pl'] = (maxmin['pl'] - 1) * -1
    maxmin = maxmin.drop('cotacao', axis=1)
    maxmin['Nota'] = maxmin.mean(axis=1) * 100

    a3 = pd.concat([a2, maxmin['Nota']], axis=1)

    a4 = a3.sort_values('Nota', ascending=False)

    print(a4)
