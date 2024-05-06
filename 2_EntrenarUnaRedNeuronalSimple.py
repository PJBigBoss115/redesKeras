import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

# Cargar los datos de Fashion MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalizar las imagenes
x_train = x_train / 255.0
x_test = x_test / 255.0

# Codificar las etiquetas en formato one-hot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Definir el modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Transformar de 28x28 a 784
    Dense(128, activation='sigmoid'),  # Primera capa oculta con 128 unidades
    Dense(64, activation='sigmoid'),  # Segunda capa oculta con 64 unidades
    Dense(10, activation='softmax')  # Capa de salida con 10 clases
])

# Compilar el modelo
model.compile(
    optimizer='sgd',  # Usar el optimizador SGD
    loss='categorical_crossentropy',  # Funcion de perdida para clasificacion
    metrics=['accuracy']  # Monitorear la precision
)

# Entrenar el modelo
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(x_test, y_test)  # Tambien podemos monitorear la precision en el conjunto de prueba
)

# Configurar el tamaño de la figura de las graficas
plt.figure(figsize=(12, 5))

# Crear un subplot para la precision
plt.subplot(1, 2, 1)
# Graficar la precision en el conjunto de entrenamiento
plt.plot(history.history['accuracy'], label='Precision (entrenamiento)')
# Graficar la precision en el conjunto de validacion
plt.plot(history.history['val_accuracy'], label='Precision (validacion)')
# Configurar el titulo del grafico
plt.title('Precision por epoca')
# Etiqueta del eje x
plt.xlabel('Epoca')
# Etiqueta del eje y
plt.ylabel('Precision')
# Mostrar la leyenda para identificar las lineas
plt.legend()

# Crear un subplot para la perdida
plt.subplot(1, 2, 2)
# Graficar la perdida en el conjunto de entrenamiento
plt.plot(history.history['loss'], label='Perdida (entrenamiento)')
# Graficar la perdida en el conjunto de validacion
plt.plot(history.history['val_loss'], label='Perdida (validacion)')
# Configurar el titulo del grafico
plt.title('Perdida por epoca')
# Etiqueta del eje x
plt.xlabel('epoca')
# Etiqueta del eje y
plt.ylabel('Perdida')
# Mostrar la leyenda para identificar las lineas
plt.legend()

# Mostrar las graficas
plt.show()

#Pregunta 3.1

# Evaluar el modelo en el conjunto de datos de prueba
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# Imprimir la precision resultante
print(f'La precision en el conjunto de prueba es: {test_accuracy:.2f}')

#Pregunta 3.2

# Realizar predicciones
predictions = model.predict(x_test)

# Convertir probabilidades a etiquetas de clase
predicted_classes = np.argmax(predictions, axis=1)

# Mostrar las primeras 10 predicciones y sus clases predichas
for i in range(10):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f'Clase predicha: {predicted_classes[i]}')
    plt.show()