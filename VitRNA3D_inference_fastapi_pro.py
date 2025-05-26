import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference_api_v2 import InferenceAPI
from VitRNA3D_inference_pro import run_VitRNA3D_inference

# 创建 API 实例
api = InferenceAPI(
    model_name="VitRNA3D",
    run_inference_func=run_VitRNA3D_inference,
    model_path="/app/model_path/VitRNA3D/RhoFold_pretrained.pt",
    data_dir="/app/datasets/VitRNA3D",
    input_webpath_key = "input_fas",  # 需要下载的文件路径在 input_content中的key
    # result_type="file",  # 重要，确保文件路径映射到宿主机
    # result_file_type = "tar.gz"
)

if __name__ == "__main__":
    api.start(port=8911)