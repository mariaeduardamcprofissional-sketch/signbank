# Treino V-Librasil + SignBank no VS Code (computador local)

Este guia assume **Windows 10/11 de 64 bits**. Onde macOS/Linux muda, está indicado.
Tudo foi testado com dados sintéticos (Python 3.12, TensorFlow 2.21 em CPU, Linux); **nada foi testado nos seus dados reais nem no Windows**.

---

## 1. O que instalar (uma vez)

| Programa | Versão | Observações |
|---|---|---|
| **Python** | **3.12** (64 bits) | Baixe em python.org/downloads. No instalador, marque **"Add python.exe to PATH"**. Evite o Python 3.13 ou mais novo: o TensorFlow pode não ter pacote para ele. |
| **Visual Studio Code** | atual | code.visualstudio.com |
| Extensão **Python** (Microsoft) | — | No VS Code: Extensões (Ctrl+Shift+X) → "Python" → Instalar. |
| Extensão **Jupyter** (opcional) | — | Só se quiser abrir o notebook `.ipynb`. |

Não precisa de GPU, CUDA, Git nem Google Drive. O TensorFlow no Windows roda em CPU, que é suficiente para este modelo (o Colab também treinava em CPU).

**Computador:** 8 GB de RAM funciona, 16 GB é confortável (o treino com SignBank usa de 2 a 4 GB no pico, estimativa minha). Reserve ~5 GB de disco livre.

Confira a instalação do Python no terminal do Windows (PowerShell): `py -3.12 --version` deve mostrar `Python 3.12.x`.

---

## 2. Montar a pasta do projeto

Use um caminho **curto, sem acentos nem espaços**, por exemplo `C:\projeto_libras`.

```
C:\projeto_libras\
  dados\
    landmarks_vlibrasil_signbank_slots.npz     <- 330 MB (o arquivo juntado)
  libras\
    __init__.py
    training_utils.py
  train_lstm_hands_face.py
  rodar_sementes.py
  treinar_tudo.py
  verificar_dados.py
  requirements.txt
```

Tudo exceto o arquivo de dados vem neste zip. **Copie para `dados\` somente o arquivo juntado de 330 MB.**

Os outros dois arquivos não são necessários para treinar:
- `landmarks_vlibrasil_trim_slots_so_vl.npz` (278 MB) só serve para a comparação "sem SignBank", que vocês dispensaram;
- `landmarks_signbank_trim.npz` (53,7 MB) já está dentro do arquivo juntado.

Guardem os dois como backup fora da pasta do projeto.

**Opcionais** (já vêm no zip, só usem se quiserem o DTW como linha de base): `knn_dtw_topk.py`, `knn_dtw_libras.py`, `diagnostico_dtw.py`, `comparar_palavra.py`. Pedem também `numba`, `scipy` e `matplotlib` (estão no `requirements.txt`).

---

## 3. Abrir no VS Code e preparar o ambiente

1. VS Code → **File → Open Folder** → `C:\projeto_libras`.
2. Abra o terminal integrado: **Ctrl+'** (aspas simples) ou *Terminal → New Terminal*. No Windows ele abre em PowerShell.
3. Crie o ambiente virtual:
   ```powershell
   py -3.12 -m venv .venv
   ```
4. Ative:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   O prompt passa a começar com `(.venv)`.
   - Se aparecer *"a execução de scripts foi desabilitada"*, rode **uma vez** `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, responda `S`, e ative de novo. Ou use o Prompt de Comando (cmd) com `.venv\Scripts\activate.bat`.
   - macOS/Linux: `python3.12 -m venv .venv` e `source .venv/bin/activate`.
5. Diga ao VS Code qual Python usar: **Ctrl+Shift+P → "Python: Select Interpreter" →** escolha o que termina em `.venv`.
6. Instale as bibliotecas (leva alguns minutos; o TensorFlow é grande):
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. Teste:
   ```powershell
   python -c "import tensorflow as tf, numpy, sklearn; print('TensorFlow', tf.__version__)"
   ```
   Mensagens sobre CUDA ou GPU não encontrada são normais e podem ser ignoradas.

---

## 4. Verificar os dados (1 minuto)

```powershell
python verificar_dados.py dados\landmarks_vlibrasil_signbank_slots.npz
```

O esperado é algo como:

```
  zip: integro
  amostras: ~4862 | classes: ~1361 | X_hands (4862, 30, 2, 21, 3) | X_traj (4862, 30, 2, 2)
  trimmed: True
  origem: signbank=~785, vlibrasil=~4077
  sinalizadores: 1=..., 2=..., 3=..., 99=~785
  mao detectada: slot0=~0.81 slot1=~0.70
  => PRONTO para treinar
```

| Se aparecer | Significa |
|---|---|
| `NAO e um .npz valido (truncado)` / `zip CORROMPIDO` | cópia incompleta; recopie o arquivo e compare o tamanho (330 MB) |
| `trimmed nao e True` | é o arquivo da extração **antiga**, sem o recorte dos quadros parados; use o gerado com `--trim` |
| `falta a chave 'X_traj'` | extração sem `--trajectory` |
| aviso de slots | os slots das mãos provavelmente **não** foram corrigidos (rodar `corrigir_slots_maos.py`) |

Para conferir o arquivo contra o original, compare o hash: `Get-FileHash dados\landmarks_vlibrasil_signbank_slots.npz` no computador de origem e aqui.

---

## 5. Treinar

Um único comando monta tudo (evita problemas de aspas do PowerShell):

```powershell
python treinar_tudo.py
```

Por padrão ele faz duas etapas, nesta ordem:

1. **Estimativa de desempenho**: 3 rodadas, um articulador de fora por vez (o SignBank fica sempre no treino). Mostra top-1, top-5 e top-10 por rodada e um resumo no final. É o **número que vai para o TCC**.
2. **Modelo final**: treina com todas as amostras (sem teste) e salva o modelo.

Para uma etapa só: `python treinar_tudo.py --etapa estimativa` ou `--etapa final`.
Para mais sementes: `--sementes 1,2,3`.

Configuração fixa: sem rosto, trajetória e velocidade, `--aug_copies 2`, 40 épocas.

**Quanto tempo?** Depende do seu processador e eu não o conheço. No Colab (CPU de 2 núcleos), uma rodada de 100 classes levava ~2,5 minutos, e eu estimei, por extrapolação, de 15 a 30 minutos por rodada no vocabulário inteiro. Para saber o seu ritmo: no treino final cada época imprime o tempo por passo (`ms/step`), e vocês multiplicam pelo número de passos por época.

**Dicas para uma execução longa:**
- Ligado na tomada, e em *Configurações → Sistema → Energia* desative o modo de suspensão durante o treino.
- Para guardar o log: `python treinar_tudo.py 2>&1 | Tee-Object -FilePath resultados\log.txt`.
- Não feche o VS Code nem o terminal. O progresso aparece rodada a rodada.

---

## 6. Resultados

A pasta `resultados\` recebe:

| Arquivo | O que é |
|---|---|
| `res_tudo.json` | top-1, top-5 e top-10 por semente |
| `proba_tudo_seed1.npz` | probabilidades por classe nas amostras de teste (para combinar com o DTW depois, sem retreinar) |
| `modelo_vlibras_signbank_final.keras` | o modelo treinado com tudo |
| `modelo_..._labels.json` | a lista de classes, na ordem do modelo |
| `modelo_..._config.json` | a representação usada (sem rosto, com trajetória e velocidade) |
| `contagem_*.csv` | amostras por classe em cada divisão |

**O modelo final não tem acurácia de teste** (treinou com tudo). O desempenho vem da etapa 1. O número que aparece como "acurácia no próprio treino" **não** é uma estimativa de desempenho.

Para as conclusões do TCC: as decisões de configuração (SignBank, correção de slots, sem rosto) foram tomadas olhando 100 classes, que fazem parte do vocabulário inteiro. O resultado fica levemente otimista para elas. Descrevam isso nas limitações e **não ajustem a configuração depois de ver este resultado**. Além disso, só ~604 das 1.361 classes têm vídeos do SignBank; o modelo tende a favorecer essas.

---

## 7. Problemas comuns

| Problema | O que fazer |
|---|---|
| `ModuleNotFoundError: No module named 'libras'` | você não está na pasta do projeto, ou a pasta `libras\` (com `__init__.py`) não foi copiada |
| `pip` não acha o `tensorflow` | o Python é novo demais (3.13+); instale o 3.12 e recrie o `.venv` |
| `MemoryError` ou o Windows fecha o programa | feche outros programas; rode com `--aug_copies 1` |
| O terminal "congela" | durante cada rodada nada é impresso; é normal. Espere o fim da rodada |
| `BadZipFile` | arquivo `.npz` incompleto; recopie (veja a seção 4) |
| Quer parar | Ctrl+C no terminal |

---

## 8. Depois do treino

- **DTW como linha de base** (opcional): instale `numba`, `scipy` (já no `requirements.txt`) e rode
  `python knn_dtw_topk.py --data dados\landmarks_vlibrasil_signbank_slots.npz --signer_regex ".*Articulador(\d+)" --so_treino 99 --w_traj 1 --penalidade 0.2 --passo_tempo 2`
  (valide o `--passo_tempo 2` em poucas classes antes, e **coloque sempre aspas** em qualquer argumento que comece com `@`, como `"@classes.txt"`).
- **Combinar DTW e LSTM**: o `proba_tudo_seed1.npz` e o `--salvar_dist` do DTW guardam o que é preciso.
