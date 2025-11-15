# Chatbot Luật (LawBot RAG)

Đây là dự án chatbot RAG hỏi đáp văn bản pháp luật, chạy 100% local (offline) bằng `PhoGPT-7B5-Instruct` và `ChromaDB`.

## 1. Yêu cầu hệ thống

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) đã được cài đặt.
* (Tùy chọn cho GPU) GPU NVIDIA (VRAM > 6GB) và driver CUDA mới nhất.

## 2. Thiết lập (Làm 1 lần)

### 1: Tải Model LLM

Dự án này sử dụng phiên bản GGUF 4-bit của PhoGPT.

1.  Tải file model (nặng ~4.37 GB) từ link sau:
    [phogpt-7b5-instruct.q4_K_m.gguf](https://huggingface.co/nguyenviet/PhoGPT-7B5-Instruct-GGUF/blob/main/phogpt-7b5-instruct.q4_K_m.gguf)
2.  Đặt file `.gguf` bạn vừa tải vào thư mục gốc của dự án này (ngang hàng với file `app.py`).

### 2: Tạo Volume cho Database

Chạy lệnh này 1 lần duy nhất để tạo nơi lưu trữ vĩnh viễn cho vector database:
```powershell
docker volume create lawbot-db
```

## 3. Hướng dẫn chạy (CPU)

Đây là phương án mặc định, chạy 100% trên CPU (chậm, 10-15 phút/câu trả lời).

### 3.1: Build Image

```powershell
docker build -t lawbot-cpu .
```

### 3.2: Nạp dữ liệu (Ingest)

Chạy `ingest.py` bên trong Docker để đọc thư mục `data/` và tạo database vào volume:
```powershell
# Dùng (^) cho PowerShell, dùng (\) cho Bash/Linux
docker run --rm -it ^
    -v ./data:/app/data:ro ^
    -v lawbot-db:/app/chroma_db ^
    lawbot-cpu python ingest.py
```

### 3.3: Chạy Chatbot

```powershell
docker run --rm -it ^
    -v ./phogpt-7b5-instruct.q4_K_m.gguf:/app/phogpt-7b5-instruct.q4_K_m.gguf:ro ^
    -v lawbot-db:/app/chroma_db ^
    lawbot-cpu
```

---

## 4. (Tùy chọn) Hướng dẫn chạy (GPU)

Phương án này yêu cầu GPU NVIDIA và nhanh hơn (vài giây/câu trả lời).

### 4.1: Sửa file `app.py` và file `requirements.txt`

Mở file `app.py`, tìm đến khối `LlamaCpp` (khoảng dòng 30) và sửa `n_gpu_layers=0` thành `n_gpu_layers=-1`.

Mở file `requirements.py`, thêm thư viện `llama-cpp-python`

### 4.2: Build Image GPU

Sử dụng file `Dockerfile.gpu` để build:
```powershell
docker build -t lawbot-gpu -f Dockerfile.gpu .
```

### 4.3: Chạy Ingest (GPU)

Giống như ingest CPU, nhưng thêm cờ `--gpus all`:
```powershell
docker run --rm -it --gpus all ^
    -v ./data:/app/data:ro ^
    -v lawbot-db:/app/chroma_db ^
    lawbot-gpu python ingest.py
```

### 4.4: Chạy Chatbot (GPU)

Giống như chạy CPU, nhưng thêm cờ `--gpus all` và dùng image `lawbot-gpu`:
```powershell
docker run --rm -it --gpus all ^
    -v ./phogpt-7b5-instruct.q4_K_m.gguf:/app/phogpt-7b5-instruct.q4_K_m.gguf:ro ^
    -v lawbot-db:/app/chroma_db ^
    lawbot-gpu
```

## 5. Hướng dẫn chạy local (CPU)

### `cd` vào folder dự án

### 5.1: Cài đặt môi trường conda

```
conda create -n lawbot python=3.11
```

### 5.2: Cài đặt thư viện

- Khởi động môi trường conda:

```
conda activate lawbot
```

- Cài thư viện `llama-cpp-python`:

```
conda install -c conda-forge llama-cpp-python
```

- Cài các thư viện còn lại:

```
pip install -r requirements.txt
```

### 5.3: Tải file `.gguf` của model qwen1_5-1.8b-chat-q8_0

```
pip install huggingface_hub
```

Sau khi cài xong thư viện huggingface_hub, chạy câu lệnh:

```
huggingface-cli download Qwen/Qwen1.5-1.8B-Chat-GGUF qwen1_5-1_8b-chat-q8_0.gguf --local-dir . --local-dir-use-symlinks False
```

### 5.4: Embedding data files

```
python ingest.py
```

### 5.5: Chạy chatbot

```
python app.py
```