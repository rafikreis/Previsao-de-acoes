import fundamentus as fd
import pandas as pd

ticker = fd.get_resultado()

x = int(input("Escolha qual tipo de ação você deseja: \n 1-Dividendos \n 2-Crescimento \n:"))

if x == 1:
    a1 = ticker.drop(['pvp', 'pcg', 'mrgliq', 'pebit', 'pacl',
                     'evebit', 'evebitda', 'mrgebit', 'psr', 'pa', 'liqc', 'patrliq'], axis=1)
    filtro = (a1['dy'] > 0.05) & (a1['divbpatr'] < 2) & (
        a1['c5y'] > 0.05) & (a1['liq2m'] > 2) & (a1['roe'] > 0.10)
    a2 = a1[filtro]

    pd.options.display.max_rows = None

    maxmin = (a2 - a2.min()) / (a2.max() - a2.min())
    maxmin['divbpatr'] = (maxmin['divbpatr'] - 1) * -1
    maxmin['pl'] = (maxmin['pl'] - 1) * -1
    maxmin = maxmin.drop('cotacao', axis=1)
    maxmin['Nota'] = maxmin.mean(axis=1) * 100

    a3 = pd.concat([a2, maxmin['Nota']], axis=1)

    a4 = a3.sort_values('Nota', ascending=False)

    print(a4)

if x == 2:
    b1 = ticker.drop(['pacl', 'pcg', 'dy', 'psr', 'pebit', 'evebit',
                     'mrgebit', 'liqc', 'psr', 'pa', 'pcg', 'pebit', 'liqc'], axis=1)

    filtro1 = (b1['pl'] > 0) & (b1['pl'] < 10) & (b1['pvp'] < 2.5) & (b1['roe'] > 0.10) & (
        b1['liq2m'] > 1) & (b1['mrgliq'] > 0.1) & (b1['c5y'] > 0.05) & (b1['divbpatr'] < 2) & (b1['roic'] > 0.10)

    b2 = b1[filtro1]

    pd.options.display.max_rows = None

    maxmin1 = (b2 - b2.min()) / (b2.max() - b2.min())
    maxmin1['pl'] = (maxmin1['pl'] - 1) * -1
    maxmin1['divbpatr'] = (maxmin1['divbpatr'] - 1) * -1
    maxmin1['pvp'] = (maxmin1['pvp'] - 1) * -1
    maxmin1['evebitda'] = (maxmin1['evebitda'] - 1) * -1

    b2['Joel Greenblatt'] = (b2['roic'] + b2['evebitda']) / 2

    b2 = b2.drop('evebitda', axis=1)

    maxmin1 = maxmin1.drop('cotacao', axis=1)
    maxmin1['Nota'] = maxmin1.mean(axis=1) * 100

    b3 = pd.concat([b2, maxmin1['Nota']], axis=1)

    b4 = b3.sort_values('Nota', ascending=False)

    print(b4)
