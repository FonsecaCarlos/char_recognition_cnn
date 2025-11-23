# char_recognition_cnn

Este projeto foi desenvolvido em novembro de 2025 para a disciplina de Visão Computacional, na Universidade Federal do Mato Grosso do Sul, Campo Grande, MS.
Foi utilizado o conjunto de imagens Chars74k para treinamento de uma CNN e depois utilizadas técnicas de OCR para reconhecimento de caracteres com base no modelo proposto.

## Este projeto foi criado em um WSL Debian.

### Para executá-lo é preciso instalar as dependências:

Atualizar o sistema

* Abra o WSL Debian e execute:
   - sudo apt update && sudo apt upgrade -y
* Instalar Python 3 + pip
  - sudo apt install -y python3 python3-pip python3-venv python3-dev
* Checar:
  - python3 --version
  - pip3 --version
* Criar um ambiente virtual (ALTAMENTE recomendado)
  - python3 -m venv mlenv
  - source mlenv/bin/activate
* Quando quiser desativar:
  - deactivate

Instalar dependências de compilação necessárias
* sudo apt install -y build-essential libssl-dev libffi-dev python3-dev \ libopenblas-dev libblas-dev liblapack-dev gfortran \ libjpeg-dev zlib1g-dev

Instalar TensorFlow no WSL Debian
* pip install tensorflow==2.20.0

Para testar:
* python3 -c "import tensorflow as tf; print(tf.__version__)"

Observação importante: O TensorFlow 2.20 roda somente em CPU no Linux via pip. Se você quiser usar GPU (CUDA) precisa instalar o TensorFlow via conda, não pip.

Instalar PyTorch + TorchVision + TorchMetrics
* pip install torch torchvision torchmetrics    // Esse comando é mais pesado, baixa as dependencias de GPU
* pip install torch torchvision torchmetrics --index-url https://download.pytorch.org/whl/cpu   // Assim baixa somente as dependencias de CPU

Para testar:
* python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"

Instalar scikit-learn, matplotlib e numpy
* pip install numpy matplotlib scikit-learn
