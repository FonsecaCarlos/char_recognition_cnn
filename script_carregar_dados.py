import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

DATASET_PATH_DEFAULT = 'chars74k_dataset_final'
IMAGE_SIZE = (64, 64) 
BATCH_SIZE = 32
NUM_CLASSES = 62    # Não é estritamente necessário se o flow_from_directory inferir

def create_generators(DATASET_PATH = DATASET_PATH_DEFAULT):

    train_datagen = ImageDataGenerator(
        # Normalização dos pixels para o intervalo [0, 1] (Dividir por 255)
        rescale=1./255,
        # Técnicas de Augmentation:
        rotation_range=10,        # Rotação aleatória (máx. 10 graus)
        width_shift_range=0.1,    # Deslocamento horizontal (até 10% da largura)
        height_shift_range=0.1,   # Deslocamento vertical (até 10% da altura)
        zoom_range=0.1,           # Zoom aleatório
        shear_range=0.1,          # Cisalhamento
        fill_mode='nearest',      # Estratégia para preencher novos pixels após transformações
        # horizontal_flip=True    # Evitado para caracteres, pois 'p' e 'd' podem se confundir
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        target_size=IMAGE_SIZE,
        color_mode='grayscale',           # CONVERTE IMAGENS PARA ESCALA DE CINZA
        batch_size=BATCH_SIZE,
        class_mode='categorical',         # Necessário para 62 classes (one-hot encoding)
        save_format='png'                 # Mantém o formato PNG (apenas durante o carregamento)
    )

    validation_generator = val_test_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'validation'),
        target_size=IMAGE_SIZE,
        color_mode='grayscale',           # CONVERTE IMAGENS PARA ESCALA DE CINZA
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'test'),
        target_size=IMAGE_SIZE,
        color_mode='grayscale',           # CONVERTE IMAGENS PARA ESCALA DE CINZA
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False                     # Desativa o shuffle para manter a ordem para avaliação final
    )

    return train_generator, validation_generator, test_generator