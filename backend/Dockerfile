FROM python:3.10.8

WORKDIR /app

COPY requirements.txt .
COPY data app/data
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
