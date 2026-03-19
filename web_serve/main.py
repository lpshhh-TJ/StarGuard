# Copyright 2026 mlq1288. Licensed under the MIT License.
# 这只是一个测试程序，在服务器正常运行时并不使用

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn

# ================= 配置区域 (请修改这里) =================
# 1. 填入华为云 ModelArts 的 DeepSeek API 地址
# 通常以 /v1/chat/completions 结尾
DEEPSEEK_API_URL = "https://api.modelarts-maas.com/v2/chat/completions"

# 2. 填入华为云 ModelArts 的 API 密钥 (Bearer Token)
DEEPSEEK_API_KEY = "yGwxmOi_Howrdi_ayyRyfjA6LbUp2jULKTgsGYUMC1EsaPhNZ1VtWWkJLYJhmKk6NB9IuDdDAiUHhxSUMl3mlQ"

# 3. 模型名称 (根据你在 ModelArts 部署的模型填写，通常是 deepseek-chat 或 deepseek-v3)
MODEL_NAME = "deepseek-v3"
# =======================================================

# 初始化 FastAPI 应用
app = FastAPI(title="DeepSeek 代理接口")

# --- 核心配置：允许跨域 (CORS) ---
# 这步非常重要，否则你的网页(端口3000)无法访问这个API(端口8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源访问
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有请求方法 (POST, GET等)
    allow_headers=["*"],  # 允许所有请求头
)

# 定义请求数据的格式模型
class ChatRequest(BaseModel):
    user_input: str

# 定义返回数据的格式模型
class ChatResponse(BaseModel):
    reply: str

# 定义 API 接口路径
@app.post("/api/deepseek-chat", response_model=ChatResponse)
async def chat_with_deepseek(request: ChatRequest):
    # 1. 构建发送给 DeepSeek 的数据包
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个智能助手。"},
            {"role": "user", "content": request.user_input}
        ],
        "temperature": 0.7  # 控制创造性，0.7 比较适中
    }

    # 2. 设置请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    # 3. 异步发送请求给华为云 ModelArts
    try:
        # 设置 timeout=60 防止模型思考时间过长导致报错
        async with httpx.AsyncClient() as client:
            response = await client.post(
                DEEPSEEK_API_URL, 
                json=payload, 
                headers=headers, 
                timeout=60.0
            )
            
            # 4. 处理返回结果
            if response.status_code == 200:
                result_json = response.json()
                # 提取 DeepSeek 返回的文本内容
                content = result_json['choices'][0]['message']['content']
                return ChatResponse(reply=content)
            else:
                # 如果 DeepSeek 服务报错
                raise HTTPException(status_code=500, detail=f"DeepSeek API 错误: {response.text}")
                
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="请求超时，DeepSeek 正在思考中...")
    except Exception as e:
        # 其他网络错误
        raise HTTPException(status_code=500, detail=str(e))

# 启动服务入口
if __name__ == "__main__":
    # 监听所有 IP (0.0.0.0)，端口 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)