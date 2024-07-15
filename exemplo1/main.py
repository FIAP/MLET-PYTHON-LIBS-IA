import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Carregando os dados a partir de um arquivo JSON
df = pd.read_json('data.json')

# Convertendo a coluna de data para o tipo datetime
df['data'] = pd.to_datetime(df['data'])

# Extraindo recursos (features) e o alvo (target)
df['data_ordinal'] = df['data'].map(pd.Timestamp.toordinal)
X = df[['data_ordinal']]
y = df['preco_fechamento']

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Root Mean Squared Error: {rmse}")

# Exibindo algumas previsões
predictions = pd.DataFrame({'Data': X_test['data_ordinal'].map(lambda x: pd.Timestamp.fromordinal(x)), 'Real': y_test, 'Predito': y_pred})
print(predictions.head())
