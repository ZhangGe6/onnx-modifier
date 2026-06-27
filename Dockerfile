FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.9-slim

WORKDIR /app

COPY . /app

RUN echo "pip install "

RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --progress-bar off

CMD ["python", "entry.py", "--host", "0.0.0.0", "--port", "5000"]
