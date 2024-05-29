import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import os

# Carregar o dataset CIFAR-10
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalizar as imagens
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Definir os nomes das classes
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Visualizar algumas imagens de treino
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i])
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

# Reduzir o conjunto de dados para acelerar o treinamento (opcional)
'''training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]'''

'''# Construir o modelo
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# Avaliar o modelo
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Salvar o modelo

model.save(model_path)
print(f"Model saved to {model_path}")'''

model_path = 'image_classifier.h5'

# Carregar o modelo salvo
model = models.load_model(model_path)

# Carregar e pré-processar a imagem
img = cv.imread('cachorro1.jpeg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (32, 32))  # Redimensionar para 32x32 pixels
img = img / 255.0  # Normalizar a imagem

# Visualizar a imagem carregada
plt.imshow(img, cmap=plt.cm.binary)
plt.show()

# Fazer a previsão
prediction = model.predict(np.array([img]))
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')


