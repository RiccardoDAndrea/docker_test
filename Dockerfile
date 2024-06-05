# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/RiccardoDAndrea/docker_test.git .

RUN pip3 install -r requirements.txt

EXPOSE 4040

HEALTHCHECK CMD curl --fail http://localhost:4040/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=4040", "--server.address=0.0.0.0"]