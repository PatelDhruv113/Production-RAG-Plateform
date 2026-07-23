FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/documents data/faiss_index data/uploads logs

# Railway automatically provides PORT
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "streamlit run src/dashboard/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true"]