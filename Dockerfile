FROM continuumio/miniconda3

WORKDIR /app

COPY requirements.txt .

# 1. Tạo môi trường Conda tên là 'lawbot' với Python 3.11
# 2. Cài đặt 'llama-cpp-python' từ kênh 'conda-forge'
RUN conda create -n lawbot python=3.11 \
    && echo "conda activate lawbot" >> ~/.bashrc \
    && conda install -n lawbot -c conda-forge llama-cpp-python -y

RUN /bin/bash -c "source activate lawbot && pip install --no-cache-dir -r requirements.txt"

COPY . .

ENTRYPOINT ["conda", "run", "-n", "lawbot"]
CMD ["python", "app.py"]