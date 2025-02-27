FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train_data.py && \
    cp house_price_model.pkl /app/house_price_model.pkl && \
    cp metrics.json /app/metrics.json

CMD ["python", "predict.py"]

