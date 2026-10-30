import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def rodar_dtw_puro_top5(caminho_proba="resultados_proba.npz", caminho_dados="landmarks_vlibrasil_trim_signbank_slots.npz"):
    data_proba = np.load(caminho_proba, allow_pickle=True)
    data_orig = np.load(caminho_dados, allow_pickle=True)
    
    classes = data_proba["classes"]
    
    if "X_hands" in data_orig:
        X_orig = data_orig["X_hands"]
    elif "X" in data_orig:
        X_orig = data_orig["X"]
    else:
        raise KeyError("Formato de landmarks não reconhecido no arquivo original.")
    
    y_orig = data_orig["y"]
    paths_orig = data_orig["paths"] if "paths" in data_orig else None
    
    chaves_proba = [k for k in data_proba.files if k.startswith("proba_")]
    
    total_amostras = 0
    acertos_lstm_top1 = 0
    acertos_dtw_puro = 0
    
    print(f"Classes carregadas: {list(classes)}")
    print("Iniciando refinamento por DTW Puro sobre o Top-5 da LSTM...\n")
    
    for chave in chaves_proba:
        fold_id = chave.replace("proba_", "")
        proba = data_proba[f"proba_{fold_id}"]
        y_real_fold = data_proba[f"y_{fold_id}"]
        paths_fold = data_proba[f"paths_{fold_id}"]
        
        print(f"=== Processando fold / sinalizador: {fold_id} ===")
        
        if paths_orig is not None:
            mask_treino = ~np.isin(paths_orig, paths_fold)
            X_treino = X_orig[mask_treino]
            y_treino = y_orig[mask_treino]
        else:
            X_treino = X_orig
            y_treino = y_orig

        for i in range(len(y_real_fold)):
            total_amostras += 1
            real_idx = y_real_fold[i]
            real_class = classes[real_idx]
            
            # Top-1 LSTM puro
            pred_lstm_idx = np.argmax(proba[i])
            if pred_lstm_idx == real_idx:
                acertos_lstm_top1 += 1
                
            # Pegar o Top-5 índices previstos pela LSTM
            top5_idx = np.argsort(proba[i])[::-1][:5]
            top5_classes = [classes[idx] for idx in top5_idx]
            
            # Encontrar a sequência correspondente ao teste atual
            if paths_orig is not None and i < len(paths_fold):
                match_idx = np.where(paths_orig == paths_fold[i])[0]
                seq_teste = X_orig[match_idx[0]].reshape(-1, X_orig.shape[-1]) if len(match_idx) > 0 else None
            else:
                seq_teste = None
                
            if seq_teste is None:
                melhor_escolha = top5_classes[0]
            else:
                seq_teste_flat = seq_teste.reshape(seq_teste.shape[0], -1)
                menor_distancia = float("inf")
                melhor_escolha = top5_classes[0]
                
                # O DTW avalia estritamente entre as 5 melhores opções da LSTM
                for cls in top5_classes:
                    amostras_cls = X_treino[y_treino == cls]
                    if len(amostras_cls) == 0:
                        continue
                    
                    amostras_cls_flat = amostras_cls.reshape(len(amostras_cls), amostras_cls.shape[1], -1)
                    for ref_seq in amostras_cls_flat:
                        try:
                            d, _ = fastdtw(seq_teste_flat, ref_seq, dist=euclidean)
                            if d < menor_distancia:
                                menor_distancia = d
                                melhor_escolha = cls
                        except Exception:
                            pass
            
            if melhor_escolha == real_class:
                acertos_dtw_puro += 1
                
    print("\n========================================")
    print(f"Total de amostras avaliadas: {total_amostras}")
    print(f"Acurácia Top-1 (LSTM Pura): {acertos_lstm_top1 / total_amostras:.4f} ({acertos_lstm_top1}/{total_amostras})")
    print(f"Acurácia Top-1 (DTW Puro Top-5): {acertos_dtw_puro / total_amostras:.4f} ({acertos_dtw_puro}/{total_amostras})")
    print("========================================")

if __name__ == "__main__":
    rodar_dtw_puro_top5()