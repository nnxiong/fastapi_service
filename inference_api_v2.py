import os
import logging
import requests
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from urllib.parse import urlparse
import copy
import datetime

def get_now_time_path_name(task_id):
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")+f"_{task_id}"


# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class InferenceAPI:
    def __init__(self, model_name: str, run_inference_func, model_path: str, data_dir: str, input_webpath_key: str):
        self.model_name = model_name
        self.run_inference_func = run_inference_func
        self.model_path = model_path
        self.data_dir = data_dir
        self.input_webpath_key = input_webpath_key


        self.host_mount_path = os.getenv("HOST_MOUNT_PATH", "/home/huangyangzhi/my_codes/models_framework_pro/models")

        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/")
        async def read_root():
            return {"message": f"Welcome to {self.model_name} Inference API!"}

        @self.app.post("/inference")
        async def inference(request: InferenceRequest):
            return self.handle_inference(request)

    def handle_inference(self, request):
        try:
            logging.info(f"Received inference request: {request}")

            input_content = copy.deepcopy(request.input_content)

            # 如果需要下载文件
            input_webpath = request.input_content.get(self.input_webpath_key)
            # if input_webpath and "http" in input_webpath:
            if input_webpath:
                # if not input_webpath:
                #     raise HTTPException(status_code=400, detail=f"{self.input_webpath_key} is required in input_content.")
                downloaded_path = self.download_file(input_webpath)
                input_content[self.input_webpath_key] = downloaded_path  # 替换路径为框架内路径

            # model_path 直接传递
            model_param_path = self.model_path

            # 处理 post output_path， 输入 的 output_path
            if request.output_path:     # 框架内位置（用于测试）
                output_path = request.output_path
            else:                       # post不给出 output path， 系统将指定框架内outputpath  e.g.   /app/datasets/test_model_name/results/...
                output_path = os.path.join(self.data_dir, f"results/{get_now_time_path_name(request.task_id)}")


            result_type, result = self.run_inference_func(input_content, request.param_dict, model_param_path, output_path)

            if isinstance(result, np.ndarray):
                result = result.tolist()

            if result_type == "file":  # 获取 当output为file时， file的宿主机路径
                docker_output_path = os.path.abspath(result)
                mount_path = self.data_dir.split("/")[1]
                host_output_path = docker_output_path.replace("/" + mount_path, self.host_mount_path)
                result = host_output_path

            response_data = {
                "status": "success",
                "task_id": request.task_id,
                "result_type": result_type,
                "result": result
            }

        except Exception as e:
            logging.error(f"Error during inference: {e}")
            response_data = {"status": "failure", "task_id": request.task_id, "detail": str(e)}

        if request.call_back_api:
            try:
                requests.post(request.call_back_api, json=response_data, timeout=10)
            except Exception as callback_error:
                logging.error(f"Error sending callback request: {callback_error}")

        return response_data

    def download_file(self, file_url: str) -> str:
        parsed_url = urlparse(file_url)
        if not parsed_url.scheme.startswith("http"):
            return file_url

        filename = os.path.basename(parsed_url.path)
        save_path = os.path.join(self.data_dir, filename)

        logging.info(f"Downloading file from {file_url} to {save_path}")
        try:
            response = requests.get(file_url, stream=True)
            response.raise_for_status()
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            logging.info(f"File downloaded successfully: {save_path}")
            return save_path
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download file: {e}")
            raise HTTPException(status_code=400, detail=f"Error downloading file: {str(e)}")

    def start(self, host="0.0.0.0", port=8000):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


class InferenceRequest(BaseModel):
    task_id: str
    call_back_api: Optional[str] = None
    input_content: Dict
    param_dict: Dict = {}
    output_path: Optional[str] = None
