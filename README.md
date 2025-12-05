# Gesture Recognition ML

Projeto exemplo para captura, preprocessamento, treino e detecção em tempo real de gestos usando webcam.

Estrutura:

- `captura.py` - coleta imagens de gestos usando webcam (salva em `dataset/<gesture>`)
- `preprocessamento.py` - carrega e prepara imagens como arrays numpy
- `modelo.py` - define e treina um modelo CNN (Keras/TensorFlow)
- `detector.py` - carrega o modelo salvo e faz inferência em tempo real
- `automacao.py` - mapeia gestos detectados para ações (usa `pyautogui`)
- `main.py` - CLI para `collect`, `train`, `detect`
- `requirements.txt` - dependências

Exemplos de uso:

Coletar imagens para um gesto:

```bash
python main.py collect thumbs_up --samples 300
```

Treinar o modelo a partir da pasta `dataset`:

```bash
python main.py train --data dataset --out model --epochs 20
```

Detectar em tempo real e executar ações mapeadas:

```bash
python main.py detect --model-dir model --threshold 0.75
```

Notas:

- Ajuste `img_size` em `preprocessamento.py` e `detector.py` se desejar resolução diferente.
- `pyautogui` pode exigir permissões; em Linux talvez seja necessário instalar dependências do sistema.
- Para um conjunto pequeno de dados, aumente `epochs` e faça augmentação manualmente.
# gesture-recognition-ml
Sistema de Interpretação de Gestos com Python, OpenCV e Machine Learning
