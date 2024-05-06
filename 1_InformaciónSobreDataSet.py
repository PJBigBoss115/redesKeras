import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

#Pregunta 1.1

# Numero de imagenes en el conjunto de entrenamiento
print(f'Numero de imagenes en el conjunto de entrenamiento: {x_train.shape[0]}')

# Numero de imagenes en el conjunto de prueba
print(f'Numero de imagenes en el conjunto de prueba: {x_test.shape[0]}')

# Tamaño de cada imagen
print(f'Tamaño de cada imagen: {x_train.shape[1]}x{x_train.shape[2]} pixeles')

#Pregunta 1.2

# Mostrar las dimensiones de los conjuntos
print("Dimensiones del conjunto de entrenamiento (imagenes):", x_train.shape)
print("Dimensiones del conjunto de prueba (imagenes):", x_test.shape)

# Mostrar un ejemplo de los datos de imagen
plt.imshow(x_train[0], cmap='gray')
plt.title(f'Etiqueta de la imagen: {y_train[0]}')
plt.colorbar()
plt.show()

# Mostrar el rango de valores de los píxeles
print("Valor minimo de pixel:", np.min(x_train))
print("Valor maximo de pixel:", np.max(x_train))

# Contar la cantidad de ejemplos por categoria en el conjunto de entrenamiento
unique, counts = np.unique(y_train, return_counts=True)
print("Numero de ejemplos por categoria en el entrenamiento:", dict(zip(unique, counts)))