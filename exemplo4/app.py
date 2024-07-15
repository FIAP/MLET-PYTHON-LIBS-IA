from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Carregar a base de dados Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converter as saídas para o formato categórico (one-hot encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Definir o modelo
model = Sequential()

# Adicionar a primeira camada (entrada)
model.add(Dense(10, input_dim=4, activation='relu'))

# Adicionar a segunda camada (oculta)
model.add(Dense(8, activation='relu'))

# Adicionar a camada de saída (3 classes)
model.add(Dense(3, activation='softmax'))

# Compilar o modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=150, batch_size=10, validation_split=0.1)

# Avaliar o modelo
_, accuracy = model.evaluate(X_test, y_test)
print(f'Acurácia: {accuracy*100:.2f}%')

# Fazer previsões
predictions = np.argmax(model.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Mapear os rótulos para os nomes das espécies
species = {0: 'Iris setosa', 1: 'Iris versicolor', 2: 'Iris virginica'}

# Exibir a primeira previsão com mais detalhes
predicted_class = predictions[0]
true_class = y_test_labels[0]
print(f'Primeira previsão: {predicted_class} ({species[predicted_class]})')
print(f'Valor Real: {true_class} ({species[true_class]})')