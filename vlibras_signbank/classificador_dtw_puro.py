import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from joblib import Parallel, delayed

def avaliar_amostra(seq_teste, templates):
    distancias = []
    for cls_idx, ref_seq in templates.items():
        try:
            d, _ = fastdtw(seq_teste, ref_seq, dist=euclidean)
            distancias.append((cls_idx, d))
        except Exception:
            pass
            
    if not distancias:
        return None
        
    # Ordenar por menor distância
    distancias.sort(key=lambda x: x[1])
    return [item[0] for item in distancias]

def rodar_classificador_dtw_global():
    caminho_dados = "landmarks_vlibrasil_trim_signbank_slots.npz"
    data = np.load(caminho_dados, allow_pickle=True)
    
    if "X_hands" in data:
        X = data["X_hands"]
    elif "X" in data:
        X = data["X"]
    else:
        raise KeyError("Chave de landmarks não encontrada no arquivo .npz")
        
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
        
    total_amostras = 0
    acertos_top1 = 0
    acertos_top5 = 0
    acertos_top10 = 0
    
    print(f"Iniciando Classificação por DTW Paralelizado (Vocabulário: {len(classes)} classes)...")
    
    for sig in sinalizadores:
        print(f"\n--- Processando Fold LOSO | Sinalizador de Teste: {sig} ---")
        mask_teste = (sign_ids == sig)
        mask_treino = ~mask_teste
        
        X_treino, y_treino = X[mask_treino], y[mask_treino]
        X_teste, y_teste = X[mask_teste], y[mask_teste]
        
        if len(X_teste) == 0:
            print(f"Aviso: Nenhuma amostra encontrada para o sinalizador {sig}. Pulando...")
            continue
            
        templates = {}
        for cls_idx in np.unique(y_treino):
            amostras_cls = X_treino[y_treino == cls_idx]
            if len(amostras_cls) > 0:
                ref_seq = amostras_cls[0].reshape(amostras_cls.shape[1], -1)
                templates[cls_idx] = ref_seq
                
        print(f"Templates criados para {len(templates)} classes. Processando teste em paralelo...")
        
        # Prepara sequências de teste
        seqs_teste_list = [X_teste[i].reshape(X_teste.shape[1], -1) for i in range(len(X_teste))]
        
        # Executa em paralelo usando todos os núcleos da CPU
        resultados_paralelos = Parallel(n_jobs=-1, verbose=5)(
            delayed(avaliar_amostra)(seq, templates) for seq in seqs_teste_list
        )
        
        for i, ranking_preds in enumerate(resultados_paralelos):
            if ranking_preds is None:
                continue
            total_amostras += 1
            real_idx = y_teste[i]
            
            if ranking_preds[0] == real_idx:
                acertos_top1 += 1
            if real_idx in ranking_preds[:5]:
                acertos_top5 += 1
            if real_idx in ranking_preds[:10]:
                acertos_top10 += 1

    print("\n========================================")
    print(f"Total de amostras avaliadas: {total_amostras}")
    print(f"Acurácia Top-1 (DTW Paralelizado):  {acertos_top1 / total_amostras:.4f} ({acertos_top1}/{total_amostras})")
    print(f"Acurácia Top-5 (DTW Paralelizado):  {acertos_top5 / total_amostras:.4f} ({acertos_top5}/{total_amostras})")
    print(f"Acurácia Top-10 (DTW Paralelizado): {acertos_top10 / total_amostras:.4f} ({acertos_top10}/{total_amostras})")
    print("========================================")

if __name__ == "__main__":
    rodar_classificador_dtw_global()