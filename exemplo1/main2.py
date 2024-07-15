import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def get_stock_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

stock_data = get_stock_data_from_file('data.json')

if 'Time Series (Daily)' in stock_data:
    timeseries = stock_data['Time Series (Daily)']
else:
    print("Erro ao obter dados do arquivo. Verifique o formato do arquivo.")
    print("Dados do arquivo:", stock_data)
    exit()

data = pd.DataFrame(timeseries).T
data = data.astype(float)
data.reset_index(inplace=True)
data.rename(columns={'index': 'date'}, inplace=True)

data['preco_futuro'] = data['4. close'].shift(-1)  
data = data.dropna()  
features = data.drop(['date', 'preco_futuro'], axis=1)
target = data['preco_futuro']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

def prever_preco(novos_dados):
    previsao = model.predict([novos_dados])
    return previsao[0]

novos_dados_exemplo = data.drop(['date', 'preco_futuro'], axis=1).iloc[-1].values  # Obter as últimas features como exemplo
previsao = prever_preco(novos_dados_exemplo)
print(f'Previsão do preço futuro: {previsao}')
