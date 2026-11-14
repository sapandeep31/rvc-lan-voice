import asyncio
import json
import logging
import os
import faulthandler
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("RVC-Server")
faulthandler.enable()  # native crashes (torch/faiss/onnx) print a Python stack instead of dying silently


def _init_model_thread():
    # faiss and torch each bundle OpenMP on macOS; multi-threaded faiss search segfaults in the
    # index lookup. omp_set_num_threads is per thread, so it is set on the one thread that runs the model.
    # Import order matters: loading faiss before voiceconversion aborts with a duplicate libomp error.
    import voiceconversion.RVC.RVCr2  # noqa: F401
    import faiss

    faiss.omp_set_num_threads(1)


# Every model call (init and inference) runs on this single thread: the model is not thread-safe,
# and it keeps faiss single-threaded no matter how many clients connect.
model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rvc-model", initializer=_init_model_thread)

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
        self.last_inference_ms = 0.0
        self.last_vol = 0.0
        self.connected_clients = 0
        # Noise-gate calibration: input volumes collected while the user stays silent
        self.calibration_vols: list[float] | None = None

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
            # downloadWeight is a coroutine; initialize() runs in a worker thread (no running loop)
            asyncio.run(downloadWeight(str(pretrain_dir)))
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

        # fcpe (PyTorch on MPS) is ~3 ms per chunk vs ~18 ms for rmvpe_onnx, which runs on CPU on macOS
        f0_det = server_cfg.get("f0_detector", "rmvpe_onnx")

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
            forceFp32=server_cfg.get("force_fp32", True),
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

        self._ensure_block_frame(audio_chunk.shape[0])

        t0 = time.perf_counter()
        result, vol, perf = self.vc.on_request(audio_chunk)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.total_processed_chunks += 1
        # Exponential moving average for latency
        self.avg_inference_ms = 0.9 * self.avg_inference_ms + 0.1 * elapsed_ms
        self.last_inference_ms = elapsed_ms
        self.last_vol = float(vol)
        if self.calibration_vols is not None:
            self.calibration_vols.append(self.last_vol)

        if self.total_processed_chunks % 100 == 0:
            logger.info(
                f"[{self.total_processed_chunks} chunks] Last inference: {elapsed_ms:.1f}ms | Avg: {self.avg_inference_ms:.1f}ms | Vol: {vol:.4f}"
            )

        return result

    def _ensure_block_frame(self, block_frame: int):
        """Resize the model's buffers to the client's frame length (client chunk_size x 128 samples).

        The library sizes its buffers once for a fixed chunk; re-allocating on change lets the
        client pick its chunk_size without a matching server restart.
        """
        if block_frame == self.vc.block_frame:
            return
        if block_frame <= 0 or block_frame % 128 != 0:
            raise ValueError(f"Frame length {block_frame} is not a positive multiple of 128 samples")
        logger.info(f"Client frame length changed: {self.vc.block_frame} -> {block_frame} samples")
        self.vc.settings.serverReadChunkSize = block_frame // 128
        self.vc.block_frame = block_frame
        self.vc.vcmodel.realloc(
            block_frame, self.vc.extra_frame, self.vc.crossfade_frame, self.vc.sola_search_frame
        )
        self.vc._generate_strength()

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
async def startup_event():
    try:
        await asyncio.get_running_loop().run_in_executor(model_executor, engine.initialize)
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
        "silent_threshold": engine.current_silent_threshold,
        "f0_detector": server_cfg.get("f0_detector", "rmvpe_onnx"),
        "frame_samples": engine.vc.block_frame if engine.vc else None,
        "avg_inference_ms": round(engine.avg_inference_ms, 1),
        "last_inference_ms": round(engine.last_inference_ms, 1),
        "input_volume": round(engine.last_vol, 4),
        "total_chunks": engine.total_processed_chunks,
        "connected_clients": engine.connected_clients,
        "calibrating": engine.calibration_vols is not None,
    }


PANEL_PATH = Path(__file__).resolve().parent / "panel.html"


@app.get("/panel")
def panel():
    return FileResponse(PANEL_PATH, media_type="text/html")


class CalibrateRequest(BaseModel):
    seconds: float = 6.0
    margin_db: float = 6.0


@app.post("/noise-gate/calibrate")
async def calibrate_noise_gate(req: CalibrateRequest):
    """Measure the room's noise floor from the live mic stream and set the gate just above it.

    The user stays silent while this runs; the gate goes to the loudest steady noise (95th
    percentile of per-chunk volume) plus margin_db, so fans/hum are muted but speech passes.
    """
    if engine.connected_clients == 0:
        raise HTTPException(status_code=409, detail="No client is streaming; start the Windows client first")
    if engine.calibration_vols is not None:
        raise HTTPException(status_code=409, detail="Calibration already running")
    engine.calibration_vols = []
    try:
        await asyncio.sleep(min(max(req.seconds, 2.0), 20.0))
        vols = engine.calibration_vols
    finally:
        engine.calibration_vols = None
    if len(vols) < 5:
        raise HTTPException(status_code=409, detail=f"Only {len(vols)} chunks arrived; is the client streaming?")
    noise = float(np.percentile(vols, 95))
    if noise < 1e-5:
        raise HTTPException(status_code=409, detail="Input is digital silence: is the mic muted or the client sending zeros?")
    noise_db = 20 * float(np.log10(noise))
    threshold = int(round(min(max(noise_db + req.margin_db, -100), -20)))
    engine.update_settings(SettingsUpdate(silent_threshold=threshold))
    logger.info(f"Noise gate calibrated over {len(vols)} chunks: floor {noise_db:.1f} dB -> gate {threshold} dB")
    return {"noise_floor_db": round(noise_db, 1), "silent_threshold": threshold, "chunks": len(vols)}


@app.post("/settings")
def set_settings(updates: SettingsUpdate):
    engine.update_settings(updates)
    return {"status": "ok", "settings": updates.dict()}


@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info(f"Client connected from {client_host} via WebSocket")
    engine.connected_clients += 1

    loop = asyncio.get_running_loop()

    try:
        while True:
            # Receive binary raw float32 PCM bytes
            raw_data = await websocket.receive_bytes()
            if not raw_data:
                continue

            audio_in = np.frombuffer(raw_data, dtype=np.float32)

            # Offload heavy synchronous GPU inference to thread pool to not block asyncio event loop
            audio_out = await loop.run_in_executor(model_executor, engine.process_chunk, audio_in)

            # Send back converted audio
            await websocket.send_bytes(audio_out.tobytes())

    except WebSocketDisconnect:
        logger.info(f"Client {client_host} disconnected cleanly.")
    except Exception as e:
        logger.error(f"WebSocket error from {client_host}: {e}")
    finally:
        engine.connected_clients -= 1


def main():
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)
    logger.info(f"Starting RVC LAN Server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
