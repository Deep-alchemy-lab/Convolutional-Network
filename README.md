RED CONVULOCIONAL 

1. IMPORT ( llamado de bibliotecas conexas) 
2. MODELO ( SECUENCIAL = UNA CAPA DETRAS DE LA OTRA)
3. LAYER ( INPUT SHAPE 3D ( H,W,C) - OUTPUT SHAPE ( H,W,C)
4. CONV 2D ( PROCESAMIENTO DE IMÁGENES)
5. POOLING ( AGRUPACIÓN DE VALORES)
6. FUNCIÓN DE ACTIVACIÓN ( UNIDAD LINEA REACTIFICADA ) RELU 
7. AÑADIR CLASIFICADOR ( FLATTEN ( aplanar o transformar a vector) DENSE ( TIPO DE CAPA ) SOFTMAX ( función de activación)
8. ENTRENAMIENTO DE LA RED ( TRAIN AND TEST)
9. EVALUAR EL MODEL ( TEST AND EVALUATE)
10. OPTIMIZADOR ( PENDIENTE DE REUBICAR EN EL ESQUEMA) 


CÓDIGO CLASIFICACIÓN BINARIA ( PERROS VS GATOS) 

# LLamando biblioteca Keras
from keras import layers
from keras import models
# Definiendo Modelo ( secuencial lineal ) 
model = models.Sequential()
# Adición de capas , procesamiento de imágenes 
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
























----------------------------------------------------------------------------------------------------------------------------------------
