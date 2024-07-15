import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Carregar os dados do dataset Reuters
max_words = 10000  # Considerar apenas as 10.000 palavras mais frequentes
max_len = 200  # Considerar apenas as primeiras 200 palavras de cada texto

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words)

# Tokenizar os dados
tokenizer = Tokenizer(num_words=max_words)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Construir o modelo
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(Dropout(0.2))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(46, activation='softmax'))

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Avaliar o modelo
score = model.evaluate(x_test, y_test, batch_size=32)
print(f'Test accuracy: {score[1]}')

# Fazer previsões
predictions = model.predict(x_test)

# Exemplo de previsão para a primeira notícia do conjunto de teste
predicted_label = predictions[0].argmax()
print(f'Predicted label: {predicted_label}')
print(f'True label: {y_test[0]}')