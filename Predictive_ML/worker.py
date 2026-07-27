"""
Container entrypoint for `ml-service` (see CONTAINERIZATION.md container #5).

Of the 25 routes under /downlink/predictive_ML/* in api_downlink.py, only these
6 actually need torch/sklearn/xgboost importable — TrainService, predict() /
predict_specific(), and load_model() (which always unpickles the stored model
object, even for a metadata-only read). The other 19 only touch Redis directly
(job status polling, model list/delete, sensor-mapping CRUD, stored-prediction
CRUD) or do lightweight telemetry/CSV work with no heavy imports, so they stay
in api_downlink.py unchanged — both containers reach the same Redis directly,
no proxying needed for those.

Training/prediction jobs use the exact same Redis job-key scheme
(train:{job_id}:..., pred:{job_id}:...) api_downlink.py already polls, so
nothing about that contract changes — only which process runs the background
work. Protected by X-Internal-Token, same pattern as docker-ops-sidecar /
backup-worker / iot-worker; end-user auth (auth.get_current_user) stays at the
api_downlink.py layer that calls this service, not here — this service has no
auth-db dependency at all, matching how Predictive_ML never touched auth/ before.
"""

import json
import logging
import uuid
from typing import Literal

import torch
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

import config
from captcha_utils import redis_client
from Predictive_ML import fetch_assets_telemetry, telemetry_processor
from Predictive_ML.ml.model_store import load_model
from Predictive_ML.ml.prediction import predict, predict_specific
from Predictive_ML.ml.train_service import TrainService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="ml-service")


def require_internal_token(x_internal_token: str = Header(default="")):
    if not config.ML_SERVICE_SHARED_SECRET or x_internal_token != config.ML_SERVICE_SHARED_SECRET:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid internal token")


class TrainModelRequest(BaseModel):
    model_name: str = Field(..., description="User-defined unique model name")
    asset_id: str
    dataset_path: str
    model_type: Literal["random_forest", "xgboost", "lstm"]
    target_column: str
    horizon: Literal["1h", "6h", "24h"]


class PredictRequest(BaseModel):
    model_name: str
    asset_id: str


class AssetFetchTrainRequest(BaseModel):
    asset_id: str
    model_name: str
    model_type: Literal["random_forest", "xgboost", "lstm"]
    target_column: str
    horizon: Literal["1h", "6h", "24h"]
    window_length: int = Field(..., gt=0, description="Window length in seconds for aggregation")


class PredictSpecificRequest(BaseModel):
    model_name: str
    asset_id: str


@app.post("/train", dependencies=[Depends(require_internal_token)])
async def submit_training_job(payload: TrainModelRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    key = f"train:{job_id}:{payload.model_name}:{payload.target_column}"

    await redis_client.set(key, json.dumps({
        "status": "queued",
        "model_name": payload.model_name,
        "target_column": payload.target_column
    }))

    async def _run():
        try:
            await redis_client.set(key, json.dumps({"status": "running"}))

            window_length = await redis_client.get(f"Window_length:{payload.asset_id}")
            freq_minutes = int(window_length) / 60 if window_length else 5.0

            train_service = TrainService()
            result = await train_service.train(
                csv_path=payload.dataset_path,
                target_column=payload.target_column,
                user_model_name=payload.model_name,
                algorithm=payload.model_type,
                horizon=payload.horizon,
                freq_minutes=freq_minutes
            )
            await redis_client.set(key, json.dumps({
                "status": "completed",
                "model_name": payload.model_name,
                "target_column": payload.target_column,
                "metrics": result["metrics"],
                "metadata": result["metadata"],
                "sensor_correlation": result["sensor_correlation"],
                "label_info": result["label_info"]
            }))
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            await redis_client.set(key, json.dumps({"status": "failed", "error": str(e)}))

    background_tasks.add_task(_run)
    return {
        "status": "accepted",
        "job_id": job_id,
        "job_key": key,
        "message": "Training started in background"
    }


@app.get("/models/{model_name}", dependencies=[Depends(require_internal_token)])
async def get_model_metadata(model_name: str):
    model, metadata = await load_model(model_name)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "status": "success",
        "model_name": model_name,
        "metadata": metadata
    }


@app.post("/predict", dependencies=[Depends(require_internal_token)])
async def predict_api(payload: PredictRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    key = f"pred:{job_id}:{payload.model_name}:{payload.asset_id}"

    await redis_client.set(key, json.dumps({
        "status": "queued",
        "model_name": payload.model_name,
        "asset_id": payload.asset_id
    }))

    async def _run():
        try:
            await redis_client.set(key, json.dumps({"status": "running"}))
            result = await predict(model_name=payload.model_name, asset_id=payload.asset_id)
            if result is None:
                await redis_client.set(key, json.dumps({
                    "status": "failed",
                    "error": "No telemetry data found"
                }))
                return
            await redis_client.set(key, json.dumps({
                "status": "completed",
                "model_name": payload.model_name,
                "asset_id": payload.asset_id,
                "result": result
            }))
        except Exception as e:
            logger.error(f"Predict job failed: {e}", exc_info=True)
            await redis_client.set(key, json.dumps({"status": "failed", "error": str(e)}))

    background_tasks.add_task(_run)
    return {
        "status": "accepted",
        "job_id": job_id,
        "job_key": key,
        "message": "Prediction started in background"
    }


@app.post("/asset-specific/fetch-train", dependencies=[Depends(require_internal_token)])
async def fetch_train_asset_model(payload: AssetFetchTrainRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    key = f"train:{job_id}:{payload.model_name}:{payload.target_column}"

    await redis_client.set(key, json.dumps({
        "status": "queued",
        "model_name": payload.model_name,
        "target_column": payload.target_column
    }))

    async def _run():
        try:
            await redis_client.set(key, json.dumps({"status": "running"}))

            telemetry_fetcher = fetch_assets_telemetry.FetchAssetsTelemetry()
            telemetry_data = telemetry_fetcher.get_telemetry_data_asset(payload.asset_id)
            if telemetry_data is None:
                await redis_client.set(key, json.dumps({
                    "status": "failed",
                    "error": "Failed to fetch telemetry data"
                }))
                return

            processor = telemetry_processor.TelemetryProcessor(telemetry_data)
            processed_data = processor.aggregate_window(window_size_sec=payload.window_length)
            processed_data = telemetry_processor.handle_missing_windows(processed_data)
            await redis_client.set(f"Window_length:{payload.asset_id}", payload.window_length)

            sensor_map_json = await redis_client.get(f"sensor_map:{payload.model_name}")
            if not sensor_map_json:
                await redis_client.set(key, json.dumps({
                    "status": "failed",
                    "error": f"Sensor mapping not found for model: {payload.model_name}"
                }))
                return

            sensor_map = json.loads(sensor_map_json)

            threshold_map = {}
            if payload.model_name == "Slipring Induction motor 60kw":
                sensor_thresholds = {
                    "Vibration_avg":      {"prefailure": 5.0,  "failure": 7.0},
                    "Temperature_avg":    {"prefailure": 80.0, "failure": 90.0},
                    "Stator_Current_avg": {"prefailure": 10.0, "failure": 15.0},
                    "Rotor_Current_avg":  {"prefailure": 8.0,  "failure": 12.0},
                }
                threshold_map = {
                    sensor_map[k]: v
                    for k, v in sensor_thresholds.items()
                    if k in sensor_map
                }

            labeled_data = telemetry_processor.label_data(
                aggregated_data=processed_data,
                threshold_map=threshold_map
            )

            train_service = TrainService()
            result = await train_service.train_specific_model(
                labeled_data=labeled_data,
                target_column=payload.target_column,
                user_model_name=payload.model_name,
                algorithm=payload.model_type,
                horizon=payload.horizon,
                equipment_type=payload.model_name,
                thresholds=threshold_map,
                freq_minutes=payload.window_length / 60,
            )

            await redis_client.set(key, json.dumps({
                "status": "completed",
                "model_name": payload.model_name,
                "target_column": payload.target_column,
                "metrics": result["metrics"],
                "metadata": result["metadata"],
                "sensor_correlation": result["sensor_correlation"],
                "label_info": result["label_info"]
            }))
        except Exception as e:
            logger.error(f"Fetch-train job failed: {e}", exc_info=True)
            await redis_client.set(key, json.dumps({"status": "failed", "error": str(e)}))

    background_tasks.add_task(_run)
    return {
        "status": "accepted",
        "job_id": job_id,
        "job_key": key,
        "message": "Fetch-train started in background"
    }


@app.post("/asset-specific/predict", dependencies=[Depends(require_internal_token)])
async def predict_specific_asset_model(payload: PredictSpecificRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    key = f"pred:{job_id}:{payload.model_name}:{payload.asset_id}"

    await redis_client.set(key, json.dumps({
        "status": "queued",
        "model_name": payload.model_name,
        "asset_id": payload.asset_id
    }))

    async def _run():
        try:
            await redis_client.set(key, json.dumps({"status": "running"}))
            result = await predict_specific(model_name=payload.model_name, asset_id=payload.asset_id)
            if result is None:
                await redis_client.set(key, json.dumps({
                    "status": "failed",
                    "error": "No telemetry data found"
                }))
                return
            await redis_client.set(key, json.dumps({
                "status": "completed",
                "model_name": payload.model_name,
                "asset_id": payload.asset_id,
                "result": result
            }))
        except Exception as e:
            logger.error(f"Asset-specific predict job failed: {e}", exc_info=True)
            await redis_client.set(key, json.dumps({"status": "failed", "error": str(e)}))

    background_tasks.add_task(_run)
    return {
        "status": "accepted",
        "job_id": job_id,
        "job_key": key,
        "message": "Asset-specific prediction started in background"
    }


@app.get("/lstm/gpu-info", dependencies=[Depends(require_internal_token)])
async def get_gpu_info():
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_info = []
            for i in range(gpu_count):
                gpu_info.append({
                    "name": torch.cuda.get_device_name(i),
                    "total_memory": torch.cuda.get_device_properties(i).total_memory,
                    "available_memory": torch.cuda.memory_allocated(i),
                    "free_memory": torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
                })
            return {
                "status": "success",
                "gpu_available": True,
                "gpu_count": gpu_count,
                "gpu_info": gpu_info
            }
        else:
            return {
                "status": "success",
                "gpu_available": False,
                "message": "No GPU available, training will use CPU which may be slower."
            }
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get GPU information")
