import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from script_carregar_dados import create_generators, IMAGE_SIZE, BATCH_SIZE
import os

# --- 1. CONFIGURAÇÕES E CARREGAMENTO DOS DADOS ---
DATASET_PATH = 'chars74k_dataset_final'
MODEL_NAME = 'char_recognition_cnn.keras'
INPUT_SHAPE = (*IMAGE_SIZE, 1) # (64, 64, 1) para imagens em escala de cinza

# 1.1. Geração dos Data Generators
print("Iniciando a criação dos Data Generators...")
train_gen, val_gen, test_gen = create_generators(DATASET_PATH)

# O número de classes é inferido pelo flow_from_directory
NUM_CLASSES = train_gen.num_classes
print(f"Número de classes inferidas: {NUM_CLASSES}")

# --- 2. DEFINIÇÃO DA ARQUITETURA DA CNN (Baseline) ---

model = Sequential([
    # Camada Convolucional 1
    Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
    MaxPooling2D((2, 2)),
    
    # Camada Convolucional 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Camada Convolucional 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flatten para alimentar a rede densa
    Flatten(),
    
    # Camada Densa 1
    Dense(512, activation='relu'),
    Dropout(0.5), # Regularização para evitar Overfitting
    
    # Camada de Saída
    Dense(NUM_CLASSES, activation='softmax') # Softmax para classificação multi-classe
])

# --- 3. COMPILAÇÃO DO MODELO ---

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', # Perda padrão para classificação multi-classe
    metrics=['accuracy']
)

model.summary()

# --- 4. CALLBACKS PARA TREINAMENTO ---

# 4.1. Model Checkpoint: Salva o melhor modelo (baseado na acurácia da validação)
checkpoint = ModelCheckpoint(
    MODEL_NAME, 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

# 4.2. Early Stopping: Para o treinamento se a métrica não melhorar após algumas épocas
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, # Espera por 5 épocas sem melhoria
    restore_best_weights=True
)

callbacks_list = [checkpoint, early_stopping]

# --- 5. TREINAMENTO DO MODELO ---

print("\nIniciando o treinamento...")

# O model.fit_generator (antigo) foi substituído por model.fit com generators
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    epochs=50, # Defina um número alto e confie no Early Stopping
    validation_data=val_gen,
    validation_steps=val_gen.samples // BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=1
)

print(f"\n✅ Treinamento concluído. O melhor modelo foi salvo como '{MODEL_NAME}'.")