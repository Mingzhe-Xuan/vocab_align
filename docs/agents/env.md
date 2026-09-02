# 环境记录

当前实现单元仅使用本地既有 Python 环境，不配置服务器虚拟环境。若后续在服务器创建环境，将使用 `python -m venv`，并在此记录创建命令、Python 与依赖版本。

## Guqq tokenizer 审计环境（2026-09-01）

- 位置：`/home/xmz/vocab_align/C2C/.venv`
- 基础解释器：`/usr/bin/python3`，Python 3.10.12（服务器无 `python` 别名，故使用等价的 `python3 -m venv`）。
- 计划创建命令：`python3 -m venv C2C/.venv`
- 计划依赖：以 editable 方式安装 `C2C` 项目；若完整项目依赖触发模型权重相关大包安装，则改为仅安装 tokenizer 审计所需的 `transformers==4.52.4` 与当前项目源码路径，不执行编译。
- 实际创建：成功，命令 `python3 -m venv C2C/.venv`。
- 安装命令：`C2C/.venv/bin/python -m pip install transformers==4.52.4`。未 editable 安装项目，因为其完整依赖会安装本次 tokenizer-only 审计不需要的 PyTorch；运行时从 `C2C` 根设置 `PYTHONPATH=.`。
- 最终直接依赖：transformers 4.52.4、tokenizers 0.21.4、huggingface-hub 0.36.2、NumPy 2.2.6、PyYAML 6.0.3、safetensors 0.8.0。
- 最终传递依赖：certifi 2026.7.22、charset-normalizer 3.5.1、filelock 3.32.5、fsspec 2026.7.0、hf-xet 1.6.0、idna 3.19、packaging 26.3、regex 2026.9.3、requests 2.34.2、tqdm 4.70.0、typing-extensions 4.16.0、urllib3 2.7.0。
- 模型框架：未安装 PyTorch/TensorFlow/Flax；该环境只允许 tokenizer/config/file utilities，不用于模型推理。
- 下载端点：`huggingface.co` 在运行时 DNS 解析失败，改用可解析的 `HF_ENDPOINT=https://hf-mirror.com` 下载锁定 revision 的 tokenizer 文件。

## Guqq sparse OT 加速依赖补充（2026-09-02）

- 复用环境：`/home/xmz/vocab_align/C2C/.venv`（Python venv，不重建、不覆盖其他任务环境）。
- 安装原因：sparse OT 的 gauge-fixed L-BFGS 加速路径延迟导入 SciPy；Job 226 提交前检查发现该 tokenizer-only venv 未安装 SciPy。
- 安装命令：`.venv/bin/python -m pip install scipy==1.15.3`；命中服务器缓存的 CPython 3.10 manylinux wheel，无源码编译。
- 验证命令：`.venv/bin/python -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)'`。
- 验证结果：NumPy `2.2.6`、SciPy `1.15.3`；与 `C2C/environment.yml` 的 SciPy 锁定版本一致。

## Guqq 阶段 3 evaluator 依赖补充（2026-09-03）

- 复用任务隔离环境：`/home/xmz/vocab_align/C2C/.venv-smoke-cu128`，该环境由 `python3 -m venv` 创建并已锁定 torch 2.7.1+cu128、transformers 4.52.4、accelerate 1.9.0；不修改共享 `.venv`。
- 预检结果：`pip show datasets` 明确未安装；统一 evaluator 顶层依赖 `datasets.load_dataset`，因此环境不完整时不得提交 Slurm。MMLU-Redux cache 目录及 `abstract_algebra` 路径已存在。
- 计划命令：`.venv-smoke-cu128/bin/python -m pip install datasets==4.0.0`。仅接受预构建 wheel；若出现源码编译或明显计算负载则停止。安装后复核 datasets、pyarrow、pandas、fsspec、huggingface-hub 与既有 torch/transformers/accelerate 版本，并运行 evaluator `--help`，模型/CUDA 不在登录节点加载。
- 实际安装成功：datasets 4.0.0、pyarrow 25.0.1、pandas 2.3.3、fsspec 2025.3.0；huggingface-hub 0.36.2、torch 2.7.1+cu128、transformers 4.52.4、accelerate 1.9.0 保持满足。全部依赖命中预构建 wheel/cache，无源码编译；fsspec 按 datasets 约束从 2026.7.0 降至 2025.3.0。
- `PYTHONPATH=. .venv-smoke-cu128/bin/python -m script.evaluation.unified_evaluator --help` 成功，仅打印既有 qwen-vl-utils 可选提示；未加载模型或初始化 CUDA。隔离环境现具备阶段 3 Slurm evaluator 入口依赖。
