import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("RVC-Server")

# Load configuration
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
else:
    config = {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "model_dir": "models",
            "pretrain_dir": "pretrain",
            "default_model": "kafka",
            "default_pitch": 7,
            "default_formant": 0.7,
            "default_index_ratio": 0.4,
            "f0_detector": "rmvpe_onnx",
            "extra_convert_size": 0.4,
            "cross_fade_overlap_size": 0.10,
            "silent_threshold": -75,
        }
    }

server_cfg = config.get("server", {})

app = FastAPI(title="RVC LAN Audio Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SettingsUpdate(BaseModel):
    pitch: Optional[int] = None
    formant_shift: Optional[float] = None
    index_ratio: Optional[float] = None
    silent_threshold: Optional[int] = None


class VoiceEngine:
    def __init__(self):
        self.vc = None
        self.device_name = "cpu"
        self.model_info = None
        self.current_pitch = server_cfg.get("default_pitch", 7)
        self.current_formant = server_cfg.get("default_formant", 0.7)
        self.current_index_ratio = server_cfg.get("default_index_ratio", 0.4)
        self.current_silent_threshold = server_cfg.get("silent_threshold", -75)
        self.total_processed_chunks = 0
        self.avg_inference_ms = 0.0

    def initialize(self):
        logger.info("Initializing voice conversion engine...")
        from voiceconversion.common.deviceManager.DeviceManager import (
            DeviceManager,
            DeviceManagerContext,
        )
        from voiceconversion.data.imported_model_info import RVCImportedModelInfo
        from voiceconversion.downloader.WeightDownloader import downloadWeight
        from voiceconversion.RVC.RVCr2 import RVCr2
        from voiceconversion.voice_changer_settings import VoiceChangerSettings
        from voiceconversion.VoiceChangerV2 import VoiceChangerV2

        # Detect best device on this machine (Mac MPS / CUDA / CPU)
        if torch.backends.mps.is_available():
            logger.info("Detected Apple Silicon GPU (Metal Performance Shaders / MPS)!")
            self.device_name = "mps"
        elif torch.cuda.is_available():
            logger.info(f"Detected NVIDIA GPU ({torch.cuda.get_device_name(0)})!")
            self.device_name = "cuda"
        else:
            logger.info("Running on CPU with multi-threaded optimization.")
            self.device_name = "cpu"

        # Determine model path
        models_dir = Path(__file__).resolve().parent.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        pretrain_dir = Path(__file__).resolve().parent.parent / "pretrain"
        if not pretrain_dir.exists():
            pretrain_dir = Path(os.path.expanduser("~/AppData/Local/pretrain"))
            if not pretrain_dir.exists():
                pretrain_dir = Path(os.path.expanduser("~/.cache/pretrain"))
                pretrain_dir.mkdir(parents=True, exist_ok=True)

        # Download pretrain weights if missing
        try:
            downloadWeight(str(pretrain_dir))
        except Exception as e:
            logger.warning(f"Could not download pretrain weights automatically: {e}")

        # Search for model weights (.safetensors or .pth)
        model_files = list(models_dir.rglob("*.safetensors")) + list(models_dir.rglob("*.pth"))
        index_files = list(models_dir.rglob("*.index"))

        model_file = str(model_files[0]) if model_files else ""
        index_file = str(index_files[0]) if index_files else ""

        if not model_file:
            # Fallback to local default if present
            default_win_model = Path(os.path.expanduser("~/AppData/Local/AVocOrg/AVoc/model_dir/0/kafka.safetensors"))
            if default_win_model.exists():
                model_file = str(default_win_model)
                index_file = str(default_win_model.parent / "added_IVF131_Flat_nprobe_1_kafka_v2.index")

        logger.info(f"Using voice model: {model_file}")
        logger.info(f"Using feature index: {index_file}")

        self.model_info = RVCImportedModelInfo(
            id=0,
            storageDir=str(Path(model_file).parent) if model_file else "",
            voiceChangerType="RVC",
            name="kafka",
            description="",
            credit="",
            termsOfUseUrl="",
            speakers={"0": "target"},
            modelFile=os.path.basename(model_file),
            modelFileOnnx="",
            indexFile=os.path.basename(index_file),
            defaultTune=self.current_pitch,
            defaultFormantShift=self.current_formant,
            defaultIndexRatio=self.current_index_ratio,
            defaultProtect=0.5,
            isONNX=False,
            modelType="pyTorchRVCv2",
            modelTypeOnnx="onnxRVC",
            samplingRate=40000,
            f0=True,
            embChannels=768,
            embOutputLayer=12,
            useFinalProj=False,
            deprecated=False,
            embedder="hubert_base",
            sampleId="",
            version="v2",
        )

        f0_det = server_cfg.get("f0_detector", "rmvpe_onnx")
        # On Mac MPS without onnxruntime-gpu, use rmvpe PyTorch or rmvpe_onnx CPU
        if self.device_name == "mps" and f0_det == "rmvpe_onnx":
            # Test if rmvpe or rmvpe_onnx is faster
            f0_det = "rmvpe_onnx"

        settings = VoiceChangerSettings(
            inputSampleRate=48000,
            outputSampleRate=48000,
            gpu=0 if self.device_name == "cuda" else -1,
            dstId=0,
            f0Detector=f0_det,
            serverReadChunkSize=56,
            extraConvertSize=server_cfg.get("extra_convert_size", 0.4),
            crossFadeOverlapSize=server_cfg.get("cross_fade_overlap_size", 0.10),
            silentThreshold=self.current_silent_threshold,
            silenceFront=1,
            forceFp32=True,
            disableJit=0,
            rvcImportedModelInfo=self.model_info,
        )

        with DeviceManagerContext():
            dm = DeviceManager.get_instance()
            dm.initialize(settings.gpu, settings.forceFp32, settings.disableJit)
            self.vc = VoiceChangerV2(settings)
            self.vc.initialize(RVCr2(settings), str(pretrain_dir))

        logger.info("Voice conversion engine successfully initialized and ready!")

    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        if self.vc is None:
            return audio_chunk

        t0 = time.perf_counter()
        result, vol, perf = self.vc.on_request(audio_chunk)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.total_processed_chunks += 1
        # Exponential moving average for latency
        self.avg_inference_ms = 0.9 * self.avg_inference_ms + 0.1 * elapsed_ms

        if self.total_processed_chunks % 100 == 0:
            logger.info(
                f"[{self.total_processed_chunks} chunks] Last inference: {elapsed_ms:.1f}ms | Avg: {self.avg_inference_ms:.1f}ms | Vol: {vol:.4f}"
            )

        return result

    def update_settings(self, updates: SettingsUpdate):
        if updates.pitch is not None:
            self.current_pitch = updates.pitch
            self.model_info.defaultTune = updates.pitch
        if updates.formant_shift is not None:
            self.current_formant = updates.formant_shift
            self.model_info.defaultFormantShift = updates.formant_shift
        if updates.index_ratio is not None:
            self.current_index_ratio = updates.index_ratio
            self.model_info.defaultIndexRatio = updates.index_ratio
        if updates.silent_threshold is not None:
            self.current_silent_threshold = updates.silent_threshold
            if self.vc and hasattr(self.vc, "vcmodel"):
                self.vc.vcmodel.inputSensitivity = 10 ** (updates.silent_threshold / 20)

        if self.vc and hasattr(self.vc, "vcmodel"):
            self.vc.vcmodel.settings.rvcImportedModelInfo = self.model_info

        logger.info(
            f"Updated settings: pitch={self.current_pitch}, formant={self.current_formant}, index={self.current_index_ratio}, threshold={self.current_silent_threshold}"
        )


engine = VoiceEngine()


@app.on_event("startup")
def startup_event():
    try:
        engine.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize voice engine at startup: {e}")
        logger.warning("Will retry when model files are present.")


@app.get("/")
def index():
    return {
        "status": "online",
        "service": "RVC LAN Voice Server",
        "device": engine.device_name,
        "model": engine.model_info.name if engine.model_info else "none",
        "avg_latency_ms": round(engine.avg_inference_ms, 1),
        "total_chunks": engine.total_processed_chunks,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": engine.device_name,
        "pitch": engine.current_pitch,
        "formant": engine.current_formant,
        "index_ratio": engine.current_index_ratio,
    }


@app.post("/settings")
def set_settings(updates: SettingsUpdate):
    engine.update_settings(updates)
    return {"status": "ok", "settings": updates.dict()}


@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info(f"Client connected from {client_host} via WebSocket")

    loop = asyncio.get_running_loop()

    try:
        while True:
            # Receive binary raw float32 PCM bytes
            raw_data = await websocket.receive_bytes()
            if not raw_data:
                continue

            audio_in = np.frombuffer(raw_data, dtype=np.float32)

            # Offload heavy synchronous GPU inference to thread pool to not block asyncio event loop
            audio_out = await loop.run_in_executor(None, engine.process_chunk, audio_in)

            # Send back converted audio
            await websocket.send_bytes(audio_out.tobytes())

    except WebSocketDisconnect:
        logger.info(f"Client {client_host} disconnected cleanly.")
    except Exception as e:
        logger.error(f"WebSocket error from {client_host}: {e}")


def main():
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)
    logger.info(f"Starting RVC LAN Server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
