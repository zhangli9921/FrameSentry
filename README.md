# FrameSentry

本地 GPU 加速的视频内容**初筛（screening）**与人工复核工具。
Local GPU-accelerated video content **pre-screening** and human review tool.

> **重要 / Important:** FrameSentry 是**初筛辅助**，不是最终审核结论，不能替代人工终审与合规流程。
> It is a first-pass aid — **not** final moderation.

## 状态 Status

M1 桌面扫描器基础：PySide6 GUI、队列扫描、NudeNet 3.4.2 适配（显式 ORT providers）、事件聚类、原生复核面板（缩略图浏览器）、拖放导入、文件日志、Windows CI。

## 目标类别 Target classes (NudeNet 3.4.2 真实标签)

- `FEMALE_BREAST_EXPOSED`
- `FEMALE_GENITALIA_EXPOSED`
- `MALE_GENITALIA_EXPOSED`
- `ANUS_EXPOSED`
- `BUTTOCKS_EXPOSED`

默认：`sample_fps=2`，`threshold=0.35`（召回优先，勿随意抬高默认阈值）。

## 环境 Environment

- **Python:** `>=3.11,<3.12`（推荐 3.11）
- **OS:** Windows 10/11（主目标）；开发可在 Linux
- **GUI:** PySide6
- **检测:** `nudenet==3.4.2` + 项目内 `NudeNetBackend`（**不** patch site-packages）

### CPU 安装 (开发 / CI)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
# 或显式：
pip install onnxruntime PySide6 opencv-python-headless numpy nudenet==3.4.2 pytest
pip install -e . --no-deps
```

### Windows NVIDIA GPU 安装（推荐）

**切勿同时安装 `onnxruntime` 与 `onnxruntime-gpu`，二者会抢占导入。**
无需安装完整 CUDA Toolkit；`onnxruntime-gpu[cuda,cudnn]` 自带所需库。不涉及任何密钥。

在仓库根目录（已激活 Python 3.11 venv）执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-gpu.ps1
```

`scripts/setup-gpu.ps1` 同时支持 **Windows PowerShell 5.1** 与 **PowerShell 7+**（以 `$LASTEXITCODE` 判定 native 命令成败，避免 5.1 把 pip 的 stderr WARNING 当成终止错误）。

脚本步骤：确认 Python 3.11 → 卸载 `onnxruntime` / `onnxruntime-gpu` → 安装 PySide6 / OpenCV / numpy → `pip install --no-deps nudenet==3.4.2` → 安装 `onnxruntime-gpu[cuda,cudnn]==1.29.0` → `pip install -e . --no-deps` → 校验 ORT 版本与 providers → **实际构造 `NudeNetBackend(device="gpu")` 并确认 session 激活 `CUDAExecutionProvider`**。

等价手动步骤见 `requirements-gpu.txt` 顶部注释。

**Windows NVIDIA GPU 必须用 `.\scripts\setup-gpu.ps1`（或 `requirements-gpu.txt` 手动等价步骤）。不要使用 `pip install .[gpu]`** — pyproject 已不再提供误导性的 `gpu` extra。

确认 GUI 中显示 “检测到 CUDA Provider：是”（仅表示 ORT `get_available_providers()` 列出了 CUDA EP，**不等于** GPU session 已成功）。真正的 CUDA session 在扫描创建检测器时验证；若 GPU 模式开启但 session 未激活 CUDA EP，该视频会 **FAILED**（禁止静默回退 CPU）。setup-gpu.ps1 在安装结束时已做一次真实 session 冒烟。

## NudeNet providers 缺陷与我们的修复

Stock NudeNet 3.4.2 在 `nudenet/nudenet.py` 中：

```python
self.onnx_session = onnxruntime.InferenceSession(
    model_path or default_320n,
    # providers=C.get_available_providers() if not providers else providers,
)
```

`providers=` **被注释掉**。即使 CUDA 可用、调用方传入 providers，官方 `NudeDetector` 仍固定走 CPU。

**本项目做法：** `src/framesentry/detectors/nudenet_backend.py` 中的 `NudeNetBackend`：

1. 定位已安装的 `320n.onnx`（`Path(nudenet.__file__).parent / "320n.onnx"`）或用户指定路径 — **不提交 onnx**
2. 自行 `InferenceSession(model, providers=...)`
3. GPU：先 `ort.preload_dlls(directory="")`（加载 pip 自带的 NVIDIA CUDA/cuDNN），再 `["CUDAExecutionProvider", "CPUExecutionProvider"]`；preload 失败 → 明确异常（不回退 CPU）。注入的 `session=` 不调用 preload。CPU 路径从不 preload。
4. 创建后检查 `session.get_providers()`；GPU 请求但无 `CUDAExecutionProvider` → 明确异常
5. 在适配器内复现 3.4.2 的 preprocess/postprocess/labels（含 stock 的 `COLOR_RGBA2BGR` + `swapRB=True`，保证 OpenCV BGR 视频帧与官方模型输入一致），边界清晰，无长期 monkeypatch

GUI **禁止**直接 `import nudenet`。

## 功能 Features (M1)

- 添加文件 / 添加文件夹 / **拖放**（文件、文件夹、混合）；递归发现；去重（Windows 大小写不敏感）
- 格式：mp4 / flv / mkv / mov / avi / webm
- 任务表：名称、路径、状态、进度、命中数、错误
- 设备 GPU/CPU；显示 ORT providers 与「检测到 CUDA Provider：是/否」（非“GPU 已可用”）；CUDA session 在扫描时硬验证
- 采样 FPS 默认 2，预设 1/2/4/8 + 手动；阈值默认 0.35
- 开始 / 取消当前 / 取消队列；GUI 日志 + 用户目录文件日志；扫描在 QThread
- 状态：WAITING / SCANNING / COMPLETED / FAILED / CANCELLED；单视频失败不中断队列；0 可用采样帧 → FAILED
- 输出：`<output_root>/<stem>.<path_id>.framesentry_review/{frames/, results.json}`（同 stem 不同路径不冲突；重扫清理该复核目录；若复核目录本身是 symlink 则拒绝清理并 FAILED）
- `results.json` **保留全部原始 hits** + events 摘要
- 标注帧：红框 + `CLASS | 0.68`（文字测宽，避免分数被裁切）
- 复核面板：DetectionEvent 缩略图浏览器、最佳帧预览、展开 raw hits、双击放大

## 布局 Layout

```
src/framesentry/
  __init__.py / __main__.py / main.py
  ui/          # PySide6（含 DnD、缩略图复核）
  core/        # types, config, timecode, events, input_discovery, validation, logging
  scanner/     # video scan worker
  detectors/   # DetectorBackend + NudeNetBackend
  storage/     # results json, frames, paths
scripts/setup-gpu.ps1
tests/
.github/workflows/windows-tests.yml
```

## 运行 / 测试

```bash
framesentry
# 或
python -m framesentry

python -m compileall -q src tests
pytest -q
```

日志文件：Windows `%LOCALAPPDATA%\FrameSentry\logs\`；Linux `~/.local/share/FrameSentry/logs/`（可用环境变量 `FRAMESENTRY_LOG_DIR` 覆盖，便于测试）。

## 限制 Limits

- 依赖本地 ORT / 驱动；GPU 需正确安装 `onnxruntime-gpu` 且卸载 CPU 包
- 抽帧采样，非逐帧精审；阈值召回优先会有误报
- 解码失败帧会跳过并记 warning；整片 0 可用采样帧则 FAILED
- 不做目录监听、Shell 扩展、便携包/exe 打包（本 PR 范围外）

## License

见仓库许可信息；NudeNet / ONNX Runtime 遵循其各自许可证。
