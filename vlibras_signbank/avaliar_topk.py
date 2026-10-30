import numpy as np

def avaliar_topk(caminho_proba="resultados_proba.npz"):
    data = np.load(caminho_proba, allow_pickle=True)
    classes = data["classes"]
    
    chaves_proba = [k for k in data.files if k.startswith("proba_")]
    
    total_amostras = 0
    acertos_top1 = 0
    acertos_top5 = 0
    acertos_top10 = 0
    
    print(f"Total de classes no vocabulário: {len(classes)}")
    print("Avaliando métricas Top-K (Top-1, Top-5, Top-10)...\n")
    
    for chave in chaves_proba:
        fold_id = chave.replace("proba_", "")
        proba = data[f"proba_{fold_id}"]
        y_real = data[f"y_{fold_id}"]
        
        for i in range(len(y_real)):
            total_amostras += 1
            real_idx = y_real[i]
            
            # Ordena os índices das probabilidades em ordem decrescente
            pred_ordenadas = np.argsort(proba[i])[::-1]
            
            # Top-1
            if pred_ordenadas[0] == real_idx:
                acertos_top1 += 1
                
            # Top-5
            if real_idx in pred_ordenadas[:5]:
                acertos_top5 += 1
                
            # Top-10
            if real_idx in pred_ordenadas[:10]:
                acertos_top10 += 1
                
    print("========================================")
    print(f"Total de amostras avaliadas: {total_amostras}")
    print(f"Acurácia Top-1:  {acertos_top1 / total_amostras:.4f} ({acertos_top1}/{total_amostras})")
    print(f"Acurácia Top-5:  {acertos_top5 / total_amostras:.4f} ({acertos_top5}/{total_amostras})")
    print(f"Acurácia Top-10: {acertos_top10 / total_amostras:.4f} ({acertos_top10}/{total_amostras})")
    print("========================================")

if __name__ == "__main__":
    avaliar_topk()