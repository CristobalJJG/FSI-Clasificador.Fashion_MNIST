# FSI-Clasificador.Fashion_MNIST 🔍
Este proyecto has salido de la clase de FSI de la ULPGC, es el código editado de MNIST cambiado para que funcione con el dataset Fashion_MNIST.
***
</br>
## Dataset Fashion_MNIST 👕

Fashion_MNIST(https://www.kaggle.com/zalando-research/fashionmnist) es un dataset de imágenes de Zalando, que consta con un conjunto de entrenamiento de 60.000 ejemplos y un conjunto de tests de 10.000 ejemplos. Una imagen se compone de 28x28 píxeles en escala de grises, asociada a una etiqueta de entre 10 clases.</br></br>
**Categorías**:</br>

0. Camiseta/top</br>
1. Pantalón</br>
2. Suéter</br>
3. Vestido</br>
4. Abrigo</br>
5. Sandalia</br>
6. Camiseta</br>
7. Zapatilla</br>
8. Bolso</br>
9. Botas bajas</br>

Composición:</br>
* Cada fila es una imagen separada.</br>
* La columna 1 es la etiqueta de la clase.</br>
* Las columnas restantes son números de píxeles(28 de ancho x 28 alto = 784 píxeles).</br>
* Cada valor es la oscuridad del píxel (1 a 255)</br>
    
Ver en Kaggle: [Fashion_MNIST](https://www.kaggle.com/zalando-research/fashionmnist). </br>
Descarga desde Kaggle: [Fashion_MNIST](https://www.kaggle.com/zalando-research/fashionmnist/download)

***
</br>

## Librerías que vamos a utilizar 📚 </br>
### Keras
Keras ([Página oficial](https://keras.io/)) es un framework de alto nivel para el Deep Learning, escrito en Python, capaz de correr TensorFlow entre otros. 

### TensorFlow
TensorFlow ([Página oficial](https://www.tensorflow.org/?hl=es-419)) es una librería desarrollada por Google Brain para sus aplicaciones de aprendizaje automático.

</br>
## Entrenamiento de la Red Neuronal 🧠 </br>
Para el entrenamiento de la Red Neuronal haremos uso del keras interno de TensorFlow, además del Fashion_MNIST. </br>

~~~
import tensorflow.keras
from tensorflow.keras.datasets import fashion_mnist
from matplotlib import pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Veamos la forma tiene x_train
print("Shape:", x_train.shape)  # 60.000 imágenes de 28x28

# Veamos una imagen cualquiera, por ejemplo, con el índice 125
image = np.array(x_train[125], dtype='float')
plt.imshow(image, cmap='gray')
plt.show()

print("Label:", y_train[125])
~~~
![Imagen](https://www.kaggleusercontent.com/kf/79940194/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..MPLlWDiYG6Ab0mWDoQmKWA.RbakXAGdYINUKlLMPpLWHznj2G9JPNmc1WUQe0XFZmPC9NOYJPiLHm70En422hvMLapl930RVyqdY86SHLXUg9tBxfewMPMFVGLc4bUjtcaPb9KIANFvIQpc1SVoP9C20rDc9kdTXwwDKC2o_Cs2qLcPMp7x-6JZ2zV802iNPAf6U_42_DX33eMaCMTs3BmLAUpmGlSvJYTi_lH_An2KRAIlHVaILYkr9_jq8dw3LtMbX0jLBPhZZAmAJgn4B0_L6lB6cah_arCMQI93ENcaB4iOoCThQckwbsN3flvEd5Zieznz-DOheGPbHVEAYJOMTCRLcZJ1CoLL0tv4pqa-uH7LS76uLjQpdymHGMuxTjfbaUicyt5ICducCWZzOuV5HFFhkrz10fKEZK33ioZe3UXiBCLqHhKqlY9WTkyRupKYNtlTfAI6l39Fz2zxCw8nHv7RKeNUgAiQYmpZOtIDqYa31xBDLWxgaN39yaVCMNkJ9iU93dF8qgOvHk373mEreR3oMK_xcJ215m8lL3lNZ8xwLHFT28eGYlW70pe9GHmEkNFVE0bSQXa4Rxkp4GOuF9sYyrX9yr6YL9MnGarSm7q_jruWzBdWPr2cBWzt7u9vxDC7v19X4PM0Lo2Zsa1Jkhf9BEEuuecvn7GqVjIj59IvWBP2j9d-GnlFHFat9is.L_UdqnKYG3gpyLpPj8PC4g/__results___files/__results___0_1.png)

</br></br>
Teniendo ya todos los datos bien colocados en x_train, y_train, x_test e y_test, pasaremos de tener matrices de 28x28 píxeles a tener un vector de 784 posiciones:
~~~
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
~~~
</br>

Pasaremos las categorías al formato one_hot:
~~~
y_train = tensorflow.keras.utils.to_categorical(y_train, 10)  # 10 clases
y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

print("Label:", y_train[125])
~~~
</br>

Crearemos las capas de neuronas, la primera será la capa de entrada, tendremos 2 capas ocultas, fully connected con la función de activación sigmoide, y una última capa de salida con la funcion de activación softmax, por lo que tendremos un vector de 10 salidas:
~~~
inputs = Input(shape=(784,))  # Capa de entrada
output_h = Dense(units=40, activation='sigmoid')(inputs)  # Capa oculta
output_h2 = Dense(units=40, activation='sigmoid')(output_h)  # Capa oculta
predictions = Dense(10, activation='softmax')(output_h3)  # Capa de salida
~~~
</br>

Crearemos el modelo:
~~~
model = Model(inputs=inputs, outputs=predictions)
~~~
</br>

Entrenamos al modelo teniendo en cuenta que:</br>
* La función de error será la media del error al cuadrado. </br>
* Un optimizador con el algoritmo de Descenso por el Gradiente Estocástico (Aleatorio)
* Y se busca tener la mejor Accuracy.
~~~
model.compile(loss='mse',
              optimizer=keras.optimizers.SGD(lr=1),
              metrics=['accuracy'])
~~~
</br>

Durante el entrenamiento de la red veremos como evoluciona el accuracy del conjunto de test, usaremos como conjunto de validación las 10.000 muestras que nos aporta el propio dataset (x_test, y_test):
* Tendremos 50 iteraciones de entrenamiento.
* Iremos cogiendo de 30 en 30 lotes.
~~~
history = model.fit(x_train, y_train, epochs=50,
        batch_size=30, validation_data=(x_test, y_test))
~~~
</br>

Al terminar podremos ver una gráfica que nos muestra tanto la precisión mientras entrena, como la precisión usando las muestras de validación:
~~~
from matplotlib import pyplot as plt 

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')

plt.title('Entrenamiento MNIST')
plt.xlabel('Épocas')
plt.legend(loc="lower right")

plt.show()
~~~
