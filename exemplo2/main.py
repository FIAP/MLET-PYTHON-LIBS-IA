from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np

# Carregar os dados do MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizar os dados
X_train = X_train / 255.0
X_test = X_test / 255.0

# Converter as labels para categóricas (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Criar o modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Avaliar o modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Fazer previsões
predictions = model.predict(X_test)

# Exemplo de previsão para a primeira imagem do conjunto de teste
print(f'Predicted label: {np.argmax(predictions[0])}')
print(f'True label: {np.argmax(y_test[0])}')


