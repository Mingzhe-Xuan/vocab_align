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

## Guqq OpenHermes 500k 物化依赖补充（2026-09-03）

- 复用环境：`/home/xmz/vocab_align/C2C/.venv`（由 `python3 -m venv` 创建，不重建、不覆盖其他任务环境）。
- 预检结果：`.venv/bin/python -c 'import datasets'` 报 `ModuleNotFoundError`；物化 Slurm 脚本明确要求 `datasets==4.0.0`，因此未提交不满足环境门禁的作业。
- 计划安装命令：`.venv/bin/python -m pip install datasets==4.0.0`。依赖安装/下载属于登录节点允许的轻量环境操作；若发生源码编译或明显计算负载则停止并改走 Slurm。
- 计划验证：打印 Python、datasets、pyarrow、huggingface-hub、fsspec、requests、tqdm、NumPy 和 SciPy 版本，并执行物化模块 import/help；实际版本与安装结果将在完成后追加，运行时仍使用 `PYTHONPATH=.` 和锁定 `HF_ENDPOINT`。
- 实际安装：`.venv/bin/python -m pip install datasets==4.0.0` 成功；所有包均来自 wheel/cache 下载，无源码编译。安装将既有 fsspec 2026.7.0 按 datasets 约束降为 2025.3.0。
- 验证结果：Python 3.10.12、datasets 4.0.0、pyarrow 25.0.1、huggingface-hub 0.36.2、fsspec 2025.3.0、requests 2.34.2、tqdm 4.70.0、NumPy 2.2.6、SciPy 1.15.3、pandas 2.3.3；`PYTHONPATH=. .venv/bin/python -m script.dataset.materialize_transport_corpus --help` 通过。
- 新增传递依赖：aiohappyeyeballs 2.7.1、aiohttp 3.14.3、aiosignal 1.4.0、async-timeout 5.0.1、attrs 26.1.0、dill 0.3.8、frozenlist 1.8.0、multidict 6.7.1、multiprocess 0.70.16、propcache 0.5.2、python-dateutil 2.9.0.post0、pytz 2026.3.post1、six 1.17.0、tzdata 2026.3、xxhash 4.0.1、yarl 1.24.5。

## Guqq 真实模型 smoke 依赖计划（2026-09-03）

- 复用环境：`/home/xmz/vocab_align/C2C/.venv`，继续使用 `python3 -m venv` 创建的既有环境，不重建、不覆盖其他任务环境。
- 计划门禁：先查询 GPU 类型/显存和 Hugging Face 缓存；再检查 `torch`、`accelerate`、`transformers`、`safetensors` 的可导入版本。真实模型要求项目锁定的 `torch==2.6.0`、`accelerate==1.9.0`、`transformers==4.52.4`。
- 若缺包，仅在登录节点执行 wheel 安装/下载并记录命令与最终版本；若出现源码编译或明显计算负载则停止。CUDA 可用性、模型加载和推理必须在 Slurm GPU allocation 内验证。
- 真实模型文件可在登录节点下载到任务约定的 Hugging Face cache，但不会在登录节点加载模型；实际缓存状态和安装结果待远端预检后追加。
- 预检结果：节点 Python venv 中 `transformers==4.52.4` 已满足，但 `torch` 与 `accelerate` 均未安装；正式 Job 240 artifact 为 39,951,267 bytes，两侧 Hugging Face 模型缓存目录均存在。集群仅暴露 `gpu:1`，GPU 型号/显存需在 Slurm allocation 内用 `nvidia-smi`/PyTorch 确认。
- 项目依赖修订：`device_map: auto` 的真实模型入口依赖 Accelerate，因此在 `pyproject.toml` 中补充精确 `accelerate==1.9.0`，与 `environment.yml` 保持一致；远端计划安装 `torch==2.6.0 accelerate==1.9.0` 的 wheel，并复核 CUDA wheel 与驱动兼容性。
- 实际安装：`C2C/.venv/bin/python -m pip install torch==2.6.0 accelerate==1.9.0` 成功，全部命中预构建 manylinux wheel/cache，无源码编译。最终直接版本为 torch 2.6.0、accelerate 1.9.0、transformers 4.52.4；torch wheel 绑定 CUDA 12.4 运行库，并安装 triton 3.2.0、cuDNN 9.1.0.70、NCCL 2.21.5 等锁定传递包。CUDA 可用性仍按规范留给 Slurm allocation 验证。
- 安装前磁盘：仓库与 cache 所在文件系统 1.8T，总使用 1.5T、可用约 252G；Qwen3-8B cache 约 16G，Mistral-Nemo cache 仅 9.1M，后者缺模型权重，需要按锁定 revision 继续下载。

## Guqq Blackwell smoke 隔离环境计划（2026-09-03）

- 原因：Job 242 确认 GPU 为 RTX 5090/Blackwell `sm_120`；共享项目 venv 的 torch 2.6.0/cu124 不含该架构。保留共享 venv 不变，创建任务专用 `/home/xmz/vocab_align/C2C/.venv-smoke-cu128`。
- 创建命令：`python3 -m venv C2C/.venv-smoke-cu128`；不使用 uv。PyTorch 依照官方 cu128 index 安装 `torch==2.7.1`，预期包版本 `2.7.1+cu128`；再精确安装 `accelerate==1.9.0 transformers==4.52.4 numpy==2.2.6 PyYAML==6.0.3 scipy==1.15.3 safetensors==0.8.0`。
- 安装属于许可的 wheel 下载/环境配置；若发生源码编译则停止。CUDA import、`get_arch_list()`/`sm_120` 和 kernel 运行验证全部放入 Slurm，不在登录节点初始化 GPU。
- smoke 使用命名 profile `blackwell-cu128`，报告必须同时记录 profile、实际 `torch==2.7.1+cu128`、compiled arches、RTX 5090 compute capability；默认 `project-cu124` profile 仍精确锁定 torch 2.6.0。
- 实际创建/安装成功：`python3 -m venv C2C/.venv-smoke-cu128`；`pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128` 得到 `torch==2.7.1+cu128`，随后精确安装 accelerate 1.9.0、transformers 4.52.4、NumPy 2.2.6、PyYAML 6.0.3、SciPy 1.15.3、safetensors 0.8.0。全部为预构建 wheel，无源码编译。
- 主要 CUDA 传递版本：cuBLAS 12.8.3.14、cuDNN 9.7.1.26、NCCL 2.26.2、Triton 3.3.1；共享 `/home/xmz/vocab_align/C2C/.venv` 未修改。实际 CUDA/compiled arch/kernel 继续由 Slurm smoke 门禁和推理验证。
## Guqq 阶段 3 evaluator 依赖补充（2026-09-03）

- 复用任务隔离环境：`/home/xmz/vocab_align/C2C/.venv-smoke-cu128`，该环境由 `python3 -m venv` 创建并已锁定 torch 2.7.1+cu128、transformers 4.52.4、accelerate 1.9.0；不修改共享 `.venv`。
- 预检结果：`pip show datasets` 明确未安装；统一 evaluator 顶层依赖 `datasets.load_dataset`，因此环境不完整时不得提交 Slurm。MMLU-Redux cache 目录及 `abstract_algebra` 路径已存在。
- 计划命令：`.venv-smoke-cu128/bin/python -m pip install datasets==4.0.0`。仅接受预构建 wheel；若出现源码编译或明显计算负载则停止。安装后复核 datasets、pyarrow、pandas、fsspec、huggingface-hub 与既有 torch/transformers/accelerate 版本，并运行 evaluator `--help`，模型/CUDA 不在登录节点加载。
- 实际安装成功：datasets 4.0.0、pyarrow 25.0.1、pandas 2.3.3、fsspec 2025.3.0；huggingface-hub 0.36.2、torch 2.7.1+cu128、transformers 4.52.4、accelerate 1.9.0 保持满足。全部依赖命中预构建 wheel/cache，无源码编译；fsspec 按 datasets 约束从 2026.7.0 降至 2025.3.0。
- `PYTHONPATH=. .venv-smoke-cu128/bin/python -m script.evaluation.unified_evaluator --help` 成功，仅打印既有 qwen-vl-utils 可选提示；未加载模型或初始化 CUDA。隔离环境现具备阶段 3 Slurm evaluator 入口依赖。
