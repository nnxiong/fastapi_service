from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import logging
import uvicorn
from playwright.sync_api import sync_playwright

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MolProbityAPI:
    def __init__(self):
        self.app = FastAPI(title="MolProbity Upload Service")
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/upload")
        async def handle_upload(request: UploadRequest):
            try:
                logger.info(f"Processing file: {request.file_path}")
                result = self.upload_to_molprobity(request.base_url, request.file_path)
                if not result:
                    raise HTTPException(status_code=400, detail="Processing failed")
                return {"status": "success", "molprob_sid": result}
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    def upload_to_molprobity(self, base_url: str, file_path: str) -> Optional[str]:
        """核心处理函数"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-dev-shm-usage"
                ]
            )
            page = browser.new_page()
            try:
                page.goto(base_url)
                page.locator("input[name='uploadFile']").set_input_files(file_path)
                
                page.wait_for_selector("h3:has-text('Summary statistics')", timeout=120000)
                molprob_sid = page.locator("input[name='MolProbSID']").get_attribute("value")
                logger.info(f"Successfully processed: {molprob_sid}")
                return molprob_sid
            except Exception as e:
                logger.error(f"Browser error: {str(e)}")
                return None
            finally:
                browser.close()

    def start(self, port: int = 8000):
        """启动服务的入口方法"""
        uvicorn.run(self.app, host="0.0.0.0", port=port)

# Pydantic模型
class UploadRequest(BaseModel):
    base_url: str
    file_path: str

# 服务启动入口
if __name__ == "__main__":
    api = MolProbityAPI()
    api.start(port=8911)  # 使用您指定的8911端口
