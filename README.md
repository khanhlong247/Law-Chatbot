# Law Chatbot (LawBot RAG)

This is a RAG chatbot project for legal documents, running 100% locally (offline) using `PhoGPT-7B5-Instruct` and `ChromaDB`.

## 1. System requirements

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed.

* (GPU optional) NVIDIA GPU (VRAM > 6GB) and latest CUDA driver.

## 2. Setup (Do it once)

### 1: Download Model LLM

This project uses the 4-bit GGUF version of PhoGPT.

1. Download the model file (weighing ~4.37 GB) from the following link:
[phogpt-7b5-instruct.q4_K_m.gguf](https://huggingface.co/nguyenviet/PhoGPT-7B5-Instruct-GGUF/blob/main/phogpt-7b5-instruct.q4_K_m.gguf)
2. Place the `.gguf` file you just downloaded into the root directory of this project (equal to the `app.py` file).

### 2: Create a Volume for the Database

Run this command once only to create a permanent storage for the database vector:
```powershell
docker volume create lawbot-db
```

## 3. Running instructions (CPU)

This is the default option, running 100% on CPU (slow, 10-15 minutes/answer).

### 3.1: Build Image

```powershell
docker build -t lawbot-cpu .
```

### 3.2: Ingest Data

Run `ingest.py` inside Docker to read the `data/` directory and create the database into the volume:
```powershell
# Use (^) for PowerShell, use (\) for Bash/Linux
docker run --rm -it ^
-v ./data:/app/data:ro ^
-v lawbot-db:/app/chroma_db ^
lawbot-cpu python ingest.py
```

### 3.3: Run Chatbot

```powershell
docker run --rm -it ^
-v ./phogpt-7b5-instruct.q4_K_m.gguf:/app/phogpt-7b5-instruct.q4_K_m.gguf:ro ^
-v lawbot-db:/app/chroma_db ^
lawbot-cpu
```

---

## 4. (Optional) Running Instructions (GPU)

This option requires an NVIDIA GPU and is faster (a few seconds/answer).

### 4.1: Edit `app.py` and `requirements.txt` files

Open the `app.py` file, find the `LlamaCpp` block (around line 30) and edit `n_gpu_layers=0` to `n_gpu_layers=-1`.

Open the `requirements.py` file, add the `llama-cpp-python` library

### 4.2: Build GPU Image

Use the `Dockerfile.gpu` file to build:
```powershell
docker build -t lawbot-gpu -f Dockerfile.gpu .

```

### 4.3: Run Ingest (GPU)

Same as CPU ingest, but add the `--gpus all` flag:
```powershell
docker run --rm -it --gpus all ^
-v ./data:/app/data:ro ^
-v lawbot-db:/app/chroma_db ^
lawbot-gpu python ingest.py
```

### 4.4: Run Chatbot (GPU)

Same as CPU ingest, but add the `--gpus all` flag and use the `lawbot-gpu` image:
```powershell
docker run --rm -it --gpus all ^
-v ./phogpt-7b5-instruct.q4_K_m.gguf:/app/phogpt-7b5-instruct.q4_K_m.gguf:ro ^
-v lawbot-db:/app/chroma_db ^
lawbot-gpu
```

## 5. Instructions for running local (CPU)

### `cd` into the project folder

### 5.1: Install the conda environment

```
conda create -n lawbot python=3.11
```

### 5.2: Install libraries

- Start the conda environment:

```
conda activate lawbot
```
- Install the library `llama-cpp-python`:

```
conda install -c conda-forge llama-cpp-python
```
- Install the remaining libraries:

```
pip install -r requirements.txt
```

### 5.3: Download the `.gguf` file of the qwen1_5-1.8b-chat-q8_0 model

```
pip install huggingface_hub
```

After installing the huggingface_hub library, run the command:

```
huggingface-cli download Qwen/Qwen1.5-1.8B-Chat-GGUF qwen1_5-1_8b-chat-q8_0.gguf --local-dir . --local-dir-use-symlinks False
```

### 5.4: Embedding data files

```
python ingest.py
```

### 5.5: Running the chatbot

```
python app.py
```