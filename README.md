RED CONVULOCIONAL 

Tener en cuenta el entrenamiento , testeo y valiudación de imágenes previo a la creación de la red 

# ESTRUCTURA BÁSICA 
IMPORT ( llamado de bibliotecas conexas) 
MODELO ( SECUENCIAL = UNA CAPA DETRAS DE LA OTRA)
LAYER ( INPUT SHAPE 3D ( H,W,C) - OUTPUT SHAPE ( H,W,C)
CONV 2D ( PROCESAMIENTO DE IMÁGENES)
POOLING ( AGRUPACIÓN DE VALORES)
FUNCIÓN DE ACTIVACIÓN ( UNIDAD LINEA REACTIFICADA ) RELU 
# OPTIMIZADOR
AÑADIR OPTIMIZADOR IMPORTADO DE KERAS 
# PROCESAMIENTO DEL DATA 
lEER LAS IMÁGENES
DECOFDIFICAR EL FORMATO JPG A RGB 
CONVERTIR A TENSORES FLOTANTES
RESCALAR VALOR DE PIXELES

7. AÑADIR CLASIFICADOR ( FLATTEN ( aplanar o transformar a vector) DENSE ( TIPO DE CAPA ) SOFTMAX ( función de activación)
8. ENTRENAMIENTO DE LA RED ( TRAIN AND TEST)
9. EVALUAR EL MODEL ( TEST AND EVALUATE)


EJEMPLO CÓDIGO CLASIFICACIÓN BINARIA ( PERROS VS GATOS) 


# ENTRENAMIENTO , VALIDACIÓN Y TESTEO DE IMÁGENES 

import os, shutil
original_dataset_dir = '/Users/fchollet/Downloads/kaggle_original_data'
base_dir = '/Users/fchollet/Downloads/cats_and_dogs_small'
os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
src = os.path.join(original_dataset_dir, fname)
dst = os.path.join(train_cats_dir, fname)
shutil.copyfile(src, dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
src = os.path.join(original_dataset_dir, fname)
dst = os.path.join(validation_cats_dir, fname)
shutil.copyfile(src, dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
src = os.path.join(original_dataset_dir, fname)
dst = os.path.join(test_cats_dir, fname)
shutil.copyfile(src, dst)
 # CONTEO DE IMÁGENES 
 
 >>> print('total training cat images:', len(os.listdir(train_cats_dir)))
total training cat images: 1000
>>> print('total training dog images:', len(os.listdir(train_dogs_dir)))
total training dog images: 1000
>>> print('total validation cat images:', len(os.listdir(validation_cats_dir)))
total validation cat images: 500
>>> print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
total validation dog images: 500
>>> print('total test cat images:', len(os.listdir(test_cats_dir)))
total test cat images: 500
>>> print('total test dog images:', len(os.listdir(test_dogs_dir)))
total test dog images: 500

# CREANDO RED CONVULOCIONAL 

from keras import layers
from keras import models
model = models.Sequential()
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

# AIDCIONANDO OPTIMIZADOR 

from keras import optimizers
model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(lr=1e-4),
metrics=['acc'])

# PROCESAMIENTO DEL DATA 

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(150, 150)
batch_size=20,
class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
validation_dir,
target_size=(150, 150),
batch_size=20,
class_mode='binary')


















----------------------------------------------------------------------------------------------------------------------------------------
