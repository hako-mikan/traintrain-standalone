import os
import sys
import base64
from io import BytesIO
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, Body, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# モジュールパスの設定
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path)

# traintrainモジュールのインポート
from traintrain.trainer import train, trainer
from modules import launch_utils

# トレーニングパラメータのモデル
class TrainParams(BaseModel):
    mode: str
    model: str
    vae: str = "None"
    network_type: str
    network_rank: int
    network_alpha: float
    lora_data_directory: str = ""
    diff_target_name: str = ""
    lora_trigger_word: str = ""
    image_size: str = "512"
    train_iterations: int
    train_batch_size: int
    train_learning_rate: float
    train_optimizer: str
    train_optimizer_settings: str = ""
    train_lr_scheduler: str
    train_lr_scheduler_settings: str = ""
    save_lora_name: str = ""
    use_gradient_checkpointing: bool = False
    network_blocks: List[str]
    orig_prompt: str = ""
    targ_prompt: str = ""
    neg_prompt: str = ""
    orig_image_base64: Optional[str] = None
    targ_image_base64: Optional[str] = None
    # 他のパラメータも必要に応じて追加
    
    class Config:
        schema_extra = {
            "example": {
                "mode": "LoRA",
                "model": "model.safetensors",
                "vae": "None",
                "network_type": "lierla",
                "network_rank": 16,
                "network_alpha": 8,
                "train_iterations": 1000,
                "train_batch_size": 2,
                "train_learning_rate": 0.0001,
                "train_optimizer": "AdamW8bit",
                "train_lr_scheduler": "cosine",
                "network_blocks": ["BASE", "IN00", "IN01", "IN02", "IN03", "IN04", "IN05", "IN06", "IN07", "IN08", "IN09", "IN10", "IN11", "M00", "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11"]
            }
        }

# 環境準備（必要に応じて）
if not launch_utils.args.skip_prepare_environment:
    launch_utils.prepare_environment()

# FastAPIアプリケーションの作成
app = FastAPI(
    title="TrainTrain API",
    description="TrainTrain API for training LoRA, iLECO, Difference, ADDifT, and Multi-ADDifT models",
    version="1.0.0"
)

# CORSミドルウェアの追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# APIエンドポイントの追加
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

# Base64エンコードされた画像をPIL Imageに変換する関数
def base64_to_image(base64_str):
    if not base64_str:
        return None
    try:
        # Base64文字列からデコード
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

# トレーニング開始APIエンドポイント
@app.post("/api/train")
async def start_training(params: TrainParams):
    try:
        # Base64エンコードされた画像をPIL Imageに変換
        orig_image = base64_to_image(params.orig_image_base64)
        targ_image = base64_to_image(params.targ_image_base64)
        
        # トレーニングを開始
        result = train.train(
            False,  # save_as_json
            params.mode,
            params.model,
            params.vae,
            params.network_type,
            params.network_rank,
            params.network_alpha,
            params.lora_data_directory,
            params.diff_target_name,
            params.lora_trigger_word,
            params.image_size,
            params.train_iterations,
            params.train_batch_size,
            params.train_learning_rate,
            params.train_optimizer,
            params.train_optimizer_settings,
            params.train_lr_scheduler,
            params.train_lr_scheduler_settings,
            params.save_lora_name,
            params.use_gradient_checkpointing,
            params.network_blocks,
            params.orig_prompt,
            params.targ_prompt,
            params.neg_prompt,
            orig_image,
            targ_image,
            # 他のパラメータも必要に応じて追加
        )
        return {"status": "success", "result": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# トレーニング停止APIエンドポイント
@app.post("/api/stop")
async def stop_training(save: bool = Body(False)):
    try:
        result = train.stop_time(save)
        return {"status": "success", "result": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# キュー追加APIエンドポイント
@app.post("/api/queue")
async def add_to_queue(params: TrainParams):
    try:
        # Base64エンコードされた画像をPIL Imageに変換
        orig_image = base64_to_image(params.orig_image_base64)
        targ_image = base64_to_image(params.targ_image_base64)
        
        # キューに追加
        result = train.queue(
            False,  # save_as_json
            params.mode,
            params.model,
            params.vae,
            params.network_type,
            params.network_rank,
            params.network_alpha,
            params.lora_data_directory,
            params.diff_target_name,
            params.lora_trigger_word,
            params.image_size,
            params.train_iterations,
            params.train_batch_size,
            params.train_learning_rate,
            params.train_optimizer,
            params.train_optimizer_settings,
            params.train_lr_scheduler,
            params.train_lr_scheduler_settings,
            params.save_lora_name,
            params.use_gradient_checkpointing,
            params.network_blocks,
            params.orig_prompt,
            params.targ_prompt,
            params.neg_prompt,
            orig_image,
            targ_image,
            # 他のパラメータも必要に応じて追加
        )
        return {"status": "success", "result": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# キュー一覧取得APIエンドポイント
@app.get("/api/queue")
async def get_queue_list():
    try:
        result = train.get_del_queue_list()
        return {"status": "success", "queue": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# キュー削除APIエンドポイント
@app.delete("/api/queue/{name}")
async def delete_queue(name: str):
    try:
        result = train.get_del_queue_list(name)
        return {"status": "success", "queue": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# 利用可能なモデル一覧取得APIエンドポイント
@app.get("/api/models")
async def get_models():
    try:
        from modules.launch_utils import args
        from traintrain.scripts.traintrain import get_models_list
        
        models = get_models_list(False)
        return {"status": "success", "models": models}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# 利用可能なVAE一覧取得APIエンドポイント
@app.get("/api/vaes")
async def get_vaes():
    try:
        from modules.launch_utils import args
        from traintrain.scripts.traintrain import get_models_list
        
        vaes = get_models_list(True)
        return {"status": "success", "vaes": vaes}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# 直接実行された場合（python main.py）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)