# FrameSentry

本地 GPU 加速的视频内容**初筛（screening）**与人工复核工具。
Local GPU-accelerated video content **pre-screening** and human review tool.

> **重要 / Important:** FrameSentry 是**初筛辅助**，不是最终审核结论，不能替代人工终审与合规流程。
> It is a first-pass aid — **not** final moderation.

## 状态 Status

M1 桌面扫描器基础：PySide6 GUI、队列扫描、NudeNet 3.4.2 适配（显式 ORT providers）、事件聚类、原生复核面板、拖放导入、Windows CI。

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

### Windows NVIDIA GPU 安装（关键）

**切勿同时安装 `onnxruntime` 与 `onnxruntime-gpu`，二者会抢占导入。**

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install -r requirements-gpu.txt
# 即：onnxruntime-gpu[cuda,cudnn]==1.29.0（或当前已验证钉扎版本）
pip install -e . --no-deps
```

确认 GUI 中 “GPU模式可用: 是”，且 ORT providers 列表含 `CUDAExecutionProvider`。若 GPU 模式开启但 session 未激活 CUDA EP，程序会**报错**（禁止静默回退 CPU）。

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
3. GPU：`["CUDAExecutionProvider", "CPUExecutionProvider"]`；CPU：`["CPUExecutionProvider"]`
4. 创建后检查 `session.get_providers()`；GPU 请求但无 `CUDAExecutionProvider` → 明确异常
5. 在适配器内复现 3.4.2 的 preprocess/postprocess/labels，边界清晰，无长期 monkeypatch

GUI **禁止**直接 `import nudenet`。

## 功能 Features (M1)

- 添加文件 / 添加文件夹 / **拖放**（文件、文件夹、混合）；递归发现；去重（Windows 大小写不敏感）
- 格式：mp4 / flv / mkv / mov / avi / webm
- 任务表：名称、路径、状态、进度、命中数、错误
- 设备 GPU/CPU；显示 ORT providers；GPU 不可用时不伪装 CUDA
- 采样 FPS 默认 2，预设 1/2/4/8 + 手动；阈值默认 0.35
- 开始 / 取消当前 / 取消队列；日志；扫描在 QThread
- 状态：WAITING / SCANNING / COMPLETED / FAILED / CANCELLED；单视频失败不中断队列
- 输出：`<output_root>/<stem>.framesentry_review/{frames/, results.json}`（默认不污染片源目录）
- `results.json` **保留全部原始 hits** + events 摘要
- 标注帧：红框 + `CLASS | 0.68`（文字测宽，避免分数被裁切）
- 复核面板：DetectionEvent 聚类（默认 2s 合并窗）、最佳帧、展开 raw hits、双击放大

## 布局 Layout

```
src/framesentry/
  __init__.py / __main__.py / main.py
  ui/          # PySide6（含 DnD）
  core/        # types, config, timecode, events, input_discovery, validation
  scanner/     # video scan worker
  detectors/   # DetectorBackend + NudeNetBackend
  storage/     # results json, frames, paths
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

## 限制 Limits

- 依赖本地 ORT / 驱动；GPU 需正确安装 `onnxruntime-gpu` 且卸载 CPU 包
- 抽帧采样，非逐帧精审；阈值召回优先会有误报
- 解码失败帧会跳过并记 warning，整片不可读则标记 FAILED
- 不做目录监听、Shell 扩展、便携包/exe 打包（本 PR 范围外）

## License

见仓库许可信息；NudeNet / ONNX Runtime 遵循其各自许可证。
