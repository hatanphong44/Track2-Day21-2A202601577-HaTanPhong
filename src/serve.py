from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

# AWS S3 Configuration
S3_BUCKET = os.environ["S3_BUCKET"]
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """Tải file model.pkl từ S3 về máy khi server khởi động."""
    s3_client = boto3.client("s3")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    s3_client.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    print("Model đã được tải xuống từ S3.")


# Gọi hàm này khi module được import (chạy khi server khởi động)
download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """Endpoint kiểm tra sức khỏe server."""
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luận.

    Đầu vào: JSON {"features": [f1, f2, ..., f12]}
    Đầu ra:  JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    pred = model.predict([req.features])[0]

    label_map = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {"prediction": int(pred), "label": label_map[int(pred)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
