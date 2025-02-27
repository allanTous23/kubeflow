FROM python:3.9

WORKDIR /app
______4

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "predict.py"]

