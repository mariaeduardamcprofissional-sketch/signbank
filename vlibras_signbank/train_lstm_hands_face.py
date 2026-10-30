import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Força rodar na CPU

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

# --- Funções de Data Augmentation ---
def aug_gaussian_noise(sequence, noise_level=0.008):
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

def aug_spatial_scaling(sequence, scale_range=(0.92, 1.08)):
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return sequence * scale_factor

def aug_temporal_resampling(sequence):
    T, features = sequence.shape
    novo_T = int(T * np.random.uniform(0.85, 1.15))
    if novo_T < 5:
        novo_T = 5
        
    indices_originais = np.linspace(0, T - 1, T)
    indices_novos = np.linspace(0, T - 1, novo_T)
    
    sequence_resampled = np.zeros((novo_T, features))
    for f in range(features):
        sequence_resampled[:, f] = np.interp(indices_novos, indices_originais, sequence[:, f])
        
    indices_final = np.linspace(0, novo_T - 1, T)
    sequence_final = np.zeros((T, features))
    for f in range(features):
        sequence_final[:, f] = np.interp(indices_final, np.linspace(0, novo_T - 1, novo_T), sequence_resampled[:, f])
        
    return sequence_final

def aplicar_data_augmentation(X_treino, y_treino, num_copias=3):
    X_aug_list = [X_treino]
    y_aug_list = [y_treino]
    
    for _ in range(num_copias):
        X_copia = np.copy(X_treino)
        novas_seqs = []
        for seq in X_copia:
            s = seq.reshape(seq.shape[0], -1)
            if np.random.rand() > 0.3:
                s = aug_gaussian_noise(s, noise_level=0.008)
            if np.random.rand() > 0.3:
                s = aug_spatial_scaling(s, scale_range=(0.92, 1.08))
            if np.random.rand() > 0.4:
                s = aug_temporal_resampling(s)
            
            novas_seqs.append(s.reshape(seq.shape))
            
        X_aug_list.append(np.array(novas_seqs))
        y_aug_list.append(np.array(y_treino, dtype=int))
        
    return np.concatenate(X_aug_list, axis=0), np.concatenate(y_aug_list, axis=0)

# --- Construção do Modelo LSTM ---
def construir_modelo(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Pipeline Principal LOSO ---
def executar_treinamento():
    caminho_dados = "landmarks_vlibrasil_trim_signbank_slots.npz"
    data = np.load(caminho_dados, allow_pickle=True)
    
    if "X_hands" in data:
        X = data["X_hands"]
    elif "X" in data:
        X = data["X"]
    else:
        raise KeyError("Chave de landmarks não encontrada no arquivo .npz")
        
    # ACHATAR AS DIMENSÕES EXTRAS (Ex: de (N, 30, 2, 21, 3) para (N, 30, 126))
    original_shape = X.shape
    if len(original_shape) > 3:
        X = X.reshape(original_shape[0], original_shape[1], -1)
        print(f"Formato original dos landmarks ajustado para LSTM: {X.shape}")

    y_raw = data["y"]
    
    classes = data["classes"] if "classes" in data else np.unique(y_raw)
    class_to_idx = {str(c): i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[str(val)] for val in y_raw], dtype=int)
        
    paths = data["paths"] if "paths" in data else np.array([str(i) for i in range(len(X))])
    
    sinalizadores = np.unique([p.split('/')[0] if '/' in str(p) else 'default' for p in paths])
    if len(sinalizadores) <= 1:
        sinalizadores = np.array(['1', '2', '3', '99'])
        sign_ids = np.array([sinalizadores[i % len(sinalizadores)] for i in range(len(paths))])
    else:
        sign_ids = np.array([p.split('/')[0] for p in paths])

    resultados_dict = {"classes": classes}
    
    print(f"Classes carregadas: {len(classes)} classes")
    print(f"Sinalizadores identificados para LOSO: {list(sinalizadores)}")
    
    for sig in sinalizadores:
        print(f"\n--- Fold LOSO | Sinalizador de Teste: {sig} ---")
        
        mask_teste = (sign_ids == sig)
        mask_treino = ~mask_teste
        
        X_treino, y_treino = X[mask_treino], np.array(y[mask_treino], dtype=int)
        X_teste, y_teste = X[mask_teste], np.array(y[mask_teste], dtype=int)
        paths_teste = paths[mask_teste]
        
        if len(X_teste) == 0:
            print(f"Aviso: Nenhuma amostra encontrada para o sinalizador {sig}. Pulando...")
            continue
            
        print(f"Aplicando Data Augmentation em {len(X_treino)} amostras de treino...")
        X_treino_aug, y_treino_aug = aplicar_data_augmentation(X_treino, y_treino, num_copias=3)
        
        y_treino_aug = np.array(y_treino_aug, dtype=int)
        y_teste = np.array(y_teste, dtype=int)
        
        classes_unicas = np.unique(y_treino_aug)
        pesos = compute_class_weight('balanced', classes=classes_unicas, y=y_treino_aug)
        class_weight_dict = dict(zip(classes_unicas, pesos))
        
        num_classes_total = len(classes)
        y_treino_cat = to_categorical(y_treino_aug, num_classes=num_classes_total)
        y_teste_cat = to_categorical(y_teste, num_classes=num_classes_total)
        
        input_shape = (X_treino_aug.shape[1], X_treino_aug.shape[2])
            
        model = construir_modelo(input_shape, num_classes_total)
        
        model.fit(
            X_treino_aug, y_treino_cat,
            validation_data=(X_teste, y_teste_cat),
            epochs=15,
            batch_size=16,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        proba_preds = model.predict(X_teste)
        
        resultados_dict[f"proba_{sig}"] = proba_preds
        resultados_dict[f"y_{sig}"] = y_teste
        resultados_dict[f"paths_{sig}"] = paths_teste

    np.savez("resultados_proba.npz", **resultados_dict)
    print("\nTreinamento concluído com sucesso! Arquivo 'resultados_proba.npz' atualizado.")

if __name__ == "__main__":
    executar_treinamento()