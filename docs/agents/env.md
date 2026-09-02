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
