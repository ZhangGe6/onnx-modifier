FROM python:3.8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get clean && apt-get update -y && apt-get upgrade -y && pip install --upgrade pip

RUN pip3 install -U onnx flask

COPY ./static /static
COPY ./templates /templates
COPY ./utils /utils
COPY ./*.py /
RUN chmod +x /app.py

CMD ["/bin/bash", "-c", "/app.py"]

