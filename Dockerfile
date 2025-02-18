# Usa la imagen oficial de Python 3.9.6
FROM python:3.9.6

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

RUN mkdir -p /app/example


# Copia todo el código del repositorio al contenedor
COPY . .


RUN python -m pip install --upgrade pip
RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

RUN python -m spacy download es_core_news_sm
RUN python -m spacy download es_core_news_sm


# Define el comando de entrada (acepta múltiples argumentos)
ENTRYPOINT ["python", "main.py"]