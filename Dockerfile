
FROM python:3.10-slim


WORKDIR /app


COPY . /app

COPY stacking_model.pkl /app/stacking_model.pkl

RUN pip install fastapi uvicorn joblib numpy pydantic scikit-learn

EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
