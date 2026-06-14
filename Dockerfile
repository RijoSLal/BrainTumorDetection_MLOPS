FROM python:3.10

WORKDIR /app

COPY  requirements.txt  .

RUN  pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]