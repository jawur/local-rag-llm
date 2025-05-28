FROM python:3.11

RUN export GGML_BLAS_VENDOR=OpenBLAS

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgomp1 \
        libgl1 \
        libsm6 \
        cmake \
        libopenblas-dev \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "-m", "rag_app.run"]
