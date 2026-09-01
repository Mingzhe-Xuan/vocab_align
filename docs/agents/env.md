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
