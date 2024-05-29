# ベースイメージとしてPythonの公式イメージを使用
FROM python:3.10-slim

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

# Pythonライブラリをインストール
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir opencv-python opencv-contrib-python
