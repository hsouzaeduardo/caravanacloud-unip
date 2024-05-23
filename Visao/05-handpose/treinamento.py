import numpy as np
import tensorflow as tf
from tensorflow import keras

# Dados simulados de pontos de referência para três gestos diferentes
# Cada ponto é representado por (x, y)
# Para simplificação, usamos apenas alguns pontos

# Gesto 1: Paz
gesture_1 = np.array([
    [0.1, 0.2], [0.15, 0.25], [0.2, 0.3], [0.25, 0.35], [0.3, 0.4], # Dedo 1
    [0.1, 0.3], [0.15, 0.35], [0.2, 0.4], [0.25, 0.45], [0.3, 0.5], # Dedo 2
])

# Gesto 2: Polegar para cima
gesture_2 = np.array([
    [0.5, 0.1], [0.55, 0.15], [0.6, 0.2], [0.65, 0.25], [0.7, 0.3], # Dedo 1
    [0.5, 0.2], [0.55, 0.25], [0.6, 0.3], [0.65, 0.35], [0.7, 0.4], # Dedo 2
])

# Gesto 3: Ok
gesture_3 = np.array([
    [0.2, 0.1], [0.25, 0.15], [0.3, 0.2], [0.35, 0.25], [0.4, 0.3], # Dedo 1
    [0.2, 0.2], [0.25, 0.25], [0.3, 0.3], [0.35, 0.35], [0.4, 0.4], # Dedo 2
])

# Labels para os gestos
labels = np.array([0, 1, 2])

# Concatenar dados para treinamento
data = np.array([gesture_1, gesture_2, gesture_3])

# Redimensionar para entrada do modelo
data = data.reshape((data.shape[0], -1))

# Embaralhar os dados
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Dividir dados em treinamento e validação
train_data = data[:2]
train_labels = labels[:2]
val_data = data[2:]
val_labels = labels[2:]

# Criar o modelo
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(data.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(train_data, train_labels, epochs=50, validation_data=(val_data, val_labels))

# Salvar o modelo
model.save('model')

# Conversão para TensorFlow.js
# !tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model model/ model_web/
