# Practice 1 后端开发文档

> 版本：1.0  
> 更新时间：2026-09-14  
> 适用目录：D:\PythonProjects\practice_1\server

## 1. 项目概述

本后端是一个基于 FastAPI 的企业设备管理与智能知识库服务，主要提供：

- 用户注册、登录、退出、资料维护和密码修改
- 设备分页查询、状态筛选、增删改、批量删除
- 设备附件上传、下载和删除
- 设备 CSV 中文导入与导出
- GLM-OCR 图片文字识别、表格识别和 Markdown 返回
- PDF/TXT 文档导入、文本切分、向量化和 pgvector 检索
- 基于 Ollama 的知识库问答和 NDJSON 流式回答

后端同时连接两个数据库：

| 数据库 | 用途 | 默认端口 |
| --- | --- | ---: |
| MySQL | 用户、设备、设备日志等业务数据 | 3306 |
| PostgreSQL + pgvector | 知识文档、文本块、图片来源和向量 | 5432 |

## 2. 技术栈

| 类别 | 技术 |
| --- | --- |
| Web 框架 | FastAPI |
| ASGI 服务 | Uvicorn |
| 请求校验 | Pydantic |
| 业务数据库 | MySQL + PyMySQL |
| 向量数据库 | PostgreSQL + pgvector + psycopg |
| 文档 Embedding | Qwen3-Embedding-0.6B + Sentence Transformers |
| OCR | GLM-OCR，布局检测使用 PP-DocLayoutV3 |
| OCR 模型服务 | 本地 Ollama 或官方 GLM-OCR API |
| 大模型问答 | Ollama /api/generate |
| 密码处理 | bcrypt |
| 测试 | pytest |

依赖清单位于 server/requirements.txt。

## 3. 目录结构

~~~text
server/
├─ main.py                         # Uvicorn 启动入口
├─ requirements.txt                # Python 依赖
├─ .env                            # 本地环境变量，不提交密钥
├─ docker-compose.yml              # pgvector 服务
├─ app/
│  ├─ app_factory.py               # 创建 FastAPI 应用并挂载路由
│  ├─ core/
│  │  ├─ config.py                 # Settings 配置
│  │  ├─ auth.py                   # 内存 Token 管理
│  │  ├─ security.py               # 密码哈希
│  │  ├─ database.py               # MySQL 连接
│  │  ├─ responses.py              # 统一成功响应
│  │  └─ exceptions.py             # 全局异常处理
│  ├─ dependencies/auth.py         # current_user 认证依赖
│  ├─ model/                       # MySQL 数据访问层
│  ├─ schems/                      # Pydantic 请求模型
│  ├─ routers/                     # HTTP API 路由层
│  ├─ knowledge/
│  │  ├─ workflow.py               # 知识库业务编排
│  │  ├─ documents.py              # PDF/TXT 提取和文本切分
│  │  ├─ embedding.py              # Qwen3 Embedding
│  │  ├─ store.py                  # pgvector 数据访问
│  │  ├─ answering.py              # Ollama 问答和流式事件
│  │  └─ qa.py                     # 问答对提取
│  └─ ocr/glm_ocr.py               # GLM-OCR 统一适配
└─ test_*.py                       # 后端测试
~~~

## 4. 应用启动流程

server/main.py 导入 create_app() 创建 FastAPI 实例，app/app_factory.py 使用统一的 /api 前缀挂载全部路由，并注册全局异常处理器。

开发模式启动：

~~~powershell
cd D:\PythonProjects\practice_1\server
python main.py
~~~

默认监听：http://127.0.0.1:8000

也可以使用 Uvicorn：

~~~powershell
cd D:\PythonProjects\practice_1\server
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
~~~

启动后可访问：

- Swagger：http://127.0.0.1:8000/docs
- OpenAPI：http://127.0.0.1:8000/openapi.json
- 健康检查：http://127.0.0.1:8000/api/health

## 5. 环境配置

配置文件为 server/.env。真实密码和 API Key 不应写入文档或提交到 Git。

### 5.1 应用和 MySQL

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| APP_NAME | Practice Server | FastAPI 应用名称 |
| DB_HOST | localhost | MySQL 地址 |
| DB_PORT | 3306 | MySQL 端口 |
| DB_USER | root | MySQL 用户 |
| DB_PASSWORD | 1234 | MySQL 密码，生产环境必须修改 |
| DB_NAME | practice_db | 业务数据库 |

### 5.2 PostgreSQL/pgvector

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| VECTOR_DB_HOST | 127.0.0.1 | PostgreSQL 地址 |
| VECTOR_DB_PORT | 5432 | PostgreSQL 端口 |
| VECTOR_DB_NAME | knowledge_db | 知识库数据库 |
| VECTOR_DB_USER | postgres | PostgreSQL 用户 |
| VECTOR_DB_PASSWORD | 1234 | PostgreSQL 密码 |

启动项目自带 pgvector：

~~~powershell
cd D:\PythonProjects\practice_1\server
docker compose up -d pgvector
~~~

容器名为 practice-pgvector，数据保存在 Docker 命名卷 pgvector_data 中。

### 5.3 Embedding 配置

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| EMBEDDING_MODEL | Qwen/Qwen3-Embedding-0.6B | 文本向量模型 |
| EMBEDDING_MODEL_SOURCE | modelscope | 模型下载来源，可切换为 huggingface |
| EMBEDDING_DEVICE | cpu | 支持 cpu 或 CUDA 设备 |
| EMBEDDING_DIMENSIONS | 1024 | 必须和模型输出维度一致 |
| EMBEDDING_MODEL_CACHE_DIR | server/.embedding_models | 本地模型缓存目录 |
| KNOWLEDGE_CHUNK_SIZE | 600 | 文本块目标长度 |
| KNOWLEDGE_CHUNK_OVERLAP | 80 | 相邻文本块重叠长度 |

模型第一次调用时延迟加载并缓存；Embedding 推理通过全局锁串行执行，降低 CPU 环境下内存峰值。

### 5.4 GLM-OCR 配置

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| GLM_OCR_MODE | selfhosted | selfhosted 调本地 Ollama，maas 调官方 API |
| GLM_OCR_MODEL | glm-ocr:latest | 本地 Ollama 模型名 |
| OLLAMA_BASE_URL | http://127.0.0.1:11434 | Ollama 地址 |
| GLM_OCR_TIMEOUT | 900 | OCR 请求超时秒数 |
| GLM_OCR_CONNECT_TIMEOUT | 300 | OCR 建立连接超时秒数 |
| GLM_OCR_MAX_PDF_PAGES | 10 | PDF OCR 最大页数，超过会拒绝 |
| GLM_OCR_LAYOUT_DEVICE | cpu | PP-DocLayoutV3 布局设备 |
| GLM_OCR_MODEL_CACHE_DIR | server/.glm_ocr_models | 布局模型缓存目录 |
| ZHIPU_API_KEY | 空 | maas 模式下的官方 API Key |
| GLM_OCR_API_URL | 官方 layout parsing 地址 | 官方 API 地址 |
| GLM_OCR_API_MODEL | glm-ocr | 官方 API 模型名 |

本地 OCR 初始化固定使用 /api/generate、ollama_generate、单工作线程和串行推理。官方模式与本地模式不会自动切换。

### 5.5 Ollama 问答配置

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| OLLAMA_BASE_URL | http://127.0.0.1:11434 | Ollama 服务地址 |
| OLLAMA_MODEL | qwen3:0.6b | 知识库回答模型 |
| OLLAMA_TIMEOUT | 180 | 问答请求超时秒数 |

## 6. 统一认证和响应约定

除健康检查、数据库 ping 和用户列表外，业务接口通常要求请求头：

~~~http
Authorization: Bearer <access_token>
~~~

Token 当前保存在后端进程内存中，服务重启后全部失效。logout 会从内存 TokenStore 中撤销当前 Token。

### 6.1 成功响应

大多数业务接口使用：

~~~json
{
  "success": true,
  "message": "操作成功",
  "data": {},
  "operator": "admin"
}
~~~

部分接口还会在顶层返回 id、user_id 或 current_user。

### 6.2 错误响应

业务异常：

~~~json
{
  "success": false,
  "message": "设备不存在"
}
~~~

参数校验失败：

~~~json
{
  "success": false,
  "message": "参数校验失败",
  "errors": []
}
~~~

常见状态码：

| 状态码 | 含义 |
| ---: | --- |
| 400 | 业务参数错误、重复数据、文件格式错误 |
| 401 | Token 缺失或用户名/密码错误 |
| 404 | 资源不存在 |
| 413 | 上传内容超过大小限制 |
| 422 | Pydantic 参数校验失败 |
| 503 | OCR、Embedding、Ollama 或向量库不可用 |

## 7. 认证接口

### POST /api/auth/register

请求：

~~~json
{
  "username": "admin",
  "password": "123456",
  "role": "admin"
}
~~~

用户名长度为 3-50，密码长度为 3-100。成功时返回顶层 user_id。

### POST /api/auth/login

请求：

~~~json
{
  "username": "admin",
  "password": "123456"
}
~~~

成功返回：

~~~json
{
  "success": true,
  "message": "登录成功",
  "data": {
    "access_token": "...",
    "token_type": "bearer",
    "user": {
      "id": 1,
      "username": "admin",
      "role": "admin",
      "created_at": "..."
    }
  }
}
~~~

### 其他认证接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | /api/auth/me | 获取当前登录用户 |
| PUT | /api/auth/profile | 修改当前用户名 |
| POST | /api/auth/change-password | 修改密码 |
| POST | /api/auth/logout | 撤销当前 Token |

修改密码请求：

~~~json
{
  "current_password": "old-password",
  "new_password": "new-password"
}
~~~

新旧密码不能相同。

## 8. 设备管理接口

### 8.1 查询设备列表

~~~http
GET /api/devices?page=1&page_size=10&search=服务器&status=active
~~~

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| page | integer | 1 | 页码，从 1 开始 |
| page_size | integer | 10 | 每页 1-100 条 |
| search | string | 空 | 按设备编号或设备名称模糊搜索 |
| status | string | 空 | active、inactive、maintenance、retired |

data 返回：

~~~json
{
  "items": [],
  "total": 100,
  "page": 1,
  "page_size": 10,
  "total_pages": 10
}
~~~

### 8.2 新建设备

POST /api/devices

~~~json
{
  "device_id": "DEV-001",
  "device_name": "办公电脑",
  "model": "型号A",
  "manufacturer": "厂商A",
  "location": "一楼机房",
  "status": "active"
}
~~~

status 可选值：active、inactive、maintenance、retired。设备编号重复时返回 400。

### 8.3 更新、删除和批量删除

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | /api/devices/{device_id} | 获取设备详情和操作日志 |
| PUT | /api/devices/{device_id} | 更新设备名称、型号、厂商、位置、状态 |
| DELETE | /api/devices/{device_id} | 删除设备记录 |
| POST | /api/devices/batch-delete | 批量删除设备 |
| GET | /api/devices/{device_id}/logs | 查询设备日志 |

批量删除请求：

~~~json
{
  "device_ids": ["DEV-001", "DEV-002"]
}
~~~

返回数据包含 deleted_count、deleted_ids、missing_ids。

### 8.4 统计、CSV 和附件

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | /api/devices/statistics | 返回总数及各状态数量 |
| GET | /api/devices/export | 导出 UTF-8 BOM 中文 CSV |
| POST | /api/devices/import | 导入 UTF-8 或 GB18030 CSV |
| POST | /api/devices/{device_id}/upload | 上传设备附件 |
| GET | /api/devices/{device_id}/download | 下载设备附件 |
| POST | /api/devices/{device_id}/delete | 删除设备附件 |

导出 CSV 表头与设备列表一致：

~~~text
设备编号,设备名称,型号,厂商,位置,状态
~~~

导入同时兼容中文表头和旧英文表头。缺少设备编号或设备名称、以及已存在的设备编号会计入 skipped_count。

## 9. OCR 接口

### POST /api/ocr/parse

使用 GLM-OCR 识别单张图片，支持 JPG/JPEG、PNG、BMP、WEBP，图片大小上限为 10 MB，使用 multipart 字段 file。

返回示例：

~~~json
{
  "success": true,
  "message": "识别完成",
  "data": {
    "markdown": "# 识别结果",
    "regions": [
      {
        "text": "设备编号",
        "label": "text",
        "page_number": 1,
        "score": null,
        "bbox": [[10, 20], [100, 20], [100, 50], [10, 50]]
      }
    ],
    "provider": "selfhosted"
  }
}
~~~

当前 OCR 不伪造置信度，score 固定为 null。表格会保留 rowspan 和 colspan，Markdown 预览使用模型返回的原始 Markdown 结构。

### POST /api/knowledge/upload-image

将图片原图、用户校正后的 Markdown、OCR 区域和向量保存到图片知识库。

multipart 字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| file | 是 | 原始图片 |
| knowledge_name | 新建时是 | 图片知识库名称，最长 255 |
| document_id | 追加时使用 | 已有图片知识库 ID |
| markdown_content | 是 | 用户确认后的 Markdown，最长 200000 |
| regions_json | 否 | OCR 区域 JSON 数组 |

不传 document_id 会新建知识库；传入已有图片知识库 ID 会追加图片。同一图片通过 SHA-256 去重。

## 10. 知识库接口

### 10.1 文档预览和导入

支持的文档格式只有 PDF 和 TXT。

| 接口 | 说明 |
| --- | --- |
| POST /api/knowledge/preview | 提取并预览文本块，不写入数据库 |
| POST /api/knowledge/upload | 提取、切分、向量化并入库 |
| GET /api/knowledge/documents | 查询知识库列表 |
| DELETE /api/knowledge/documents/{document_id} | 删除知识库及其向量和原图 |

文档大小上限为 20 MB，PDF 最多 10 页。TXT 优先使用 UTF-8/UTF-8 BOM，失败后尝试 GB18030。

导入字段：

- file：PDF 或 TXT 文件
- qa_split：是否将文本块转换为问答对
- document_id：可选；填写后追加到已有知识库

PDF 解析策略：

1. 优先读取 PDF 原生文本层。
2. 如果存在扫描页，再调用 GLM-OCR。
3. 页面文本按章节、编号条目和句子边界切分。
4. 使用 Qwen3 Embedding 生成 1024 维向量。
5. 在 PostgreSQL pgvector 中事务性保存文档、文本块和向量。

### 10.2 文本块和图片来源

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | /api/knowledge/documents/{document_id}/chunks | 分页获取文本块 |
| GET | /api/knowledge/documents/{document_id}/source-image | 获取第一张来源图片 |
| GET | /api/knowledge/documents/{document_id}/images | 获取图片列表 |
| GET | /api/knowledge/documents/{document_id}/images/{image_id} | 获取指定原图 |

文本块查询参数：page 默认 1，page_size 默认 50，最大 100。

### 10.3 向量检索

POST /api/knowledge/search

~~~json
{
  "query": "设备维护周期是什么？",
  "top_k": 5
}
~~~

query 长度为 1-1000，top_k 范围为 1-20。服务端会生成 query embedding、执行 pgvector 余弦检索、对正文精确命中重新排序、拼接相邻文本块上下文；问题包含“图片中”等词时优先图片 OCR 来源。

### 10.4 知识库问答

非流式接口：POST /api/knowledge/ask。

请求格式与检索相同，data 中返回 answer、sources 和 items。

流式接口：POST /api/knowledge/ask/stream，响应类型为 application/x-ndjson：

~~~json
{"type":"metadata","sources":[],"items":[]}
{"type":"delta","content":"根据知识库内容"}
{"type":"delta","content":"，设备应……"}
{"type":"done"}
~~~

异常事件：

~~~json
{"type":"error","message":"无法连接 Ollama，请确认服务已启动"}
~~~

问答提示词要求模型只能依据检索上下文回答，不足时返回“知识库中没有足够信息”，并隐藏思考过程。

## 11. 数据模型概览

### 11.1 MySQL 业务库

主要表：

- users：用户账号、角色、密码哈希
- devices：设备信息和附件路径
- 设备日志表：设备创建、修改、删除、导入导出、附件操作记录

Device 数据访问层负责分页查询、模糊搜索、状态统计、批量删除和附件路径更新。

### 11.2 PostgreSQL 知识库

knowledge.store.ensure_schema() 会按需创建或校验：

- knowledge_documents：知识库主体、处理模式、来源类型、文件元数据、块数量
- knowledge_chunks：文本块、页码、展示文本、向量
- knowledge_images：图片原图、OCR Markdown、区域坐标、内容哈希

向量列使用 vector(EMBEDDING_DIMENSIONS)，并创建 HNSW 余弦距离索引。删除文档依赖外键级联删除文本块，同时仅删除上传目录内的原图文件。

## 12. 测试和验证

安装依赖后运行：

~~~powershell
cd D:\PythonProjects\practice_1\server
pytest -q
~~~

主要测试范围：

| 文件 | 覆盖内容 |
| --- | --- |
| test_auth_profile_support.py | 编辑资料、修改密码 |
| test_csv_encoding.py | 中文 CSV 编码和导入导出 |
| test_knowledge_api.py | 知识库 API |
| test_knowledge_llm.py | Ollama 问答和流式回答 |
| test_knowledge_model_cache.py | Embedding 模型缓存 |
| test_knowledge_support.py | 文档和图片知识库支持 |
| test_ocr_support.py | GLM-OCR 适配和错误处理 |
| test_table_structure.py | 表格 HTML、合并行列和结构化处理 |

不建议在测试中调用真实生产数据库、真实 Ollama 或官方 OCR API。优先使用 monkeypatch、临时目录和模拟模型输出。

## 13. 新功能开发规范

推荐按以下顺序实现后端功能：

1. 在 app/schems 增加请求模型和边界校验。
2. 在 app/model 或 app/knowledge/store.py 增加数据访问方法。
3. 在 app/knowledge/workflow.py 编排跨模块业务逻辑。
4. 在 app/routers 增加薄路由，只负责鉴权、输入读取、线程池调用和响应转换。
5. 复用 success_response 和 AppException，不要在每个路由中重新设计响应格式。
6. 为成功、参数错误、资源不存在、外部服务不可用分别补充测试。
7. 所有新增函数和非显然分支添加简洁中文注释。

阻塞型工作应使用 run_in_threadpool，尤其是 Embedding、GLM-OCR、PDF 解析、pgvector 查询和 Ollama 请求。

## 14. 常见故障排查

### 14.1 500 或 503，无法连接 Ollama

检查：

~~~powershell
ollama list
ollama ps
Invoke-WebRequest http://127.0.0.1:11434/api/tags
~~~

确认 .env 中 OLLAMA_BASE_URL 和 OLLAMA_MODEL 与本机一致。

### 14.2 GLM-OCR 首次识别超时

首次调用可能同时加载 Ollama 模型和 PP-DocLayoutV3 布局模型。检查 GLM_OCR_TIMEOUT、GLM_OCR_CONNECT_TIMEOUT，并确认 server/.glm_ocr_models 中模型已缓存。

### 14.3 PDF 导入失败

- 确认文件小于 20 MB。
- 确认页数不超过 10 页。
- 文本型 PDF 优先检查是否有可复制文本。
- 扫描型 PDF 检查 GLM-OCR 是否可用。
- 如果返回“未提取到文字”，先在 OCR 页面识别后再保存到图片知识库。

### 14.4 向量检索失败

检查 PostgreSQL 是否启动、数据库是否存在、pgvector 扩展是否可用，以及 EMBEDDING_DIMENSIONS 是否仍为 1024。模型输出维度不匹配会被后端主动拒绝。

### 14.5 中文 CSV 乱码

后端导出使用 UTF-8 BOM，Excel 应直接识别中文。导入支持 UTF-8/UTF-8 BOM 和 GB18030。

## 15. 当前已知限制

- TokenStore 当前为进程内存实现，不适合多进程共享登录态。
- GLM-OCR 本地模式按单工作线程串行推理，适合开发和低并发场景。
- PDF OCR 当前最多 10 页，超出必须拆分文件。
- 设备附件接口当前以单个文件路径保存，后续如需多附件应拆出附件表。
- 用户列表接口 /api/users 当前未接入登录依赖，生产环境需要根据权限模型补充鉴权。
- 设备附件删除使用 POST /devices/{device_id}/delete，设备记录删除使用 DELETE /devices/{device_id}，客户端必须使用正确 HTTP 方法。
- 生产环境应关闭 reload=True，使用反向代理、HTTPS、持久化 Token/Session 和密钥管理服务。

