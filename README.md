# 服饰以图搜商品（本地双模型原型）

这是一个完全在本机运行的服饰商品图片检索项目，适用于服装、鞋、包和饰品。系统优先返回高置信“同款”，没有可靠同款时再返回“相似款”。原始商品图片、向量索引和搜索结果不会上传到第三方服务。

## 系统怎么判断

- **pHash**：发现完全相同、重新压缩或轻微修改的图片。
- **DINOv2 ViT-B/14**：关注轮廓、结构、纹理和局部细节，主要帮助判断同款。
- **Marqo FashionSigLIP**：关注品类、颜色、材质和整体风格，主要帮助寻找相似款。
- **Qdrant Local**：在本机保存每张展示图的 `dino` 与 `fashion` 两套 768 维向量。
- **逻辑回归校准器**：利用目录里的商品 ID 自动学习同款概率，优先保证同款结果准确。
- **Gradio**：提供本地上传图片和查看结果的网页。

更详细的原理见 [docs/architecture.md](docs/architecture.md)。

## 已按你的目录配置

程序只扫描：

```text
/Users/wanglei/work/simarket/data/images/{店铺ID}/{商品ID}/display/
```

`detail/` 详情长图不会进入索引。商品唯一标识为 `店铺ID:商品ID`，程序不会修改原图。

## 环境要求

- macOS Apple Silicon（当前按 M1 Pro 16 GB 配置）
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- 首次下载模型约需 1.2 GB，建索引还会占用额外磁盘空间

## 从零开始运行

```bash
cd /Users/wanglei/work/image-search

# 1. 安装 Python 依赖，不会下载模型权重
uv sync --group dev

# 2. 检查图片目录，不加载模型
uv run fashion-search inspect

# 3. 首次下载两套模型并建立索引
uv run fashion-search index

# 4. 用同商品的多张图片校准“同款概率”
uv run fashion-search calibrate

# 5. 自动比较两个模型和多种融合权重
uv run fashion-search evaluate

# 6. 启动本地网页
uv run fashion-search serve
```

如果只想先验证完整链路，可以先索引少量图片：

```bash
uv run fashion-search index --limit 100
```

试跑不会删除 Qdrant 中其他已存在的数据。确认效果后，再运行不带 `--limit` 的全量命令。

小规模索引后也可以只评测少量查询图：

```bash
uv run fashion-search evaluate --limit 10
```

评测只会使用已经写入 Qdrant 的图片，不会因为原图目录更大而自动执行全量推理。

浏览器打开：<http://127.0.0.1:7860>

停止服务时在终端按 `Ctrl-C`。

## 为什么安装依赖时没有立刻下载模型

依赖和模型是两类文件：

1. `uv sync` 安装 PyTorch、Qdrant、Gradio 等程序运行库。
2. 模型权重只在显式运行模型测试或 `fashion-search index` 时下载。

这样执行 `inspect`、单元测试或阅读代码时不会被迫下载 1 GB 以上的文件，也避免把权重误放进 Git 仓库。权重来自公开 Hugging Face 模型仓库，不需要 Token，并固定到以下 revision：

```text
Marqo/marqo-fashionSigLIP
c56244cc94f92419e8369fa71efdaf403b124ce8

facebook/dinov2-base
f9e44c814b77203eaa57a6bdbbd535f21ede1415
```

如果文件已手动放入以下目录，程序会优先离线加载，不再重复下载：

```text
state/models/marqo-fashionSigLIP/open_clip_model.safetensors
state/models/dinov2-base/model.safetensors
state/models/dinov2-base/config.json
state/models/dinov2-base/preprocessor_config.json
```

否则模型会进入 Hugging Face 的本机共享缓存；`.gitignore` 同时排除了虚拟环境与整个 `state/` 运行目录。

## 常用开发命令

```bash
# 普通测试：不会下载或执行真实模型
uv run pytest -m "not model" -v

# 真实模型冒烟测试：首次运行会下载权重
uv run pytest tests/test_encoders.py -m model -v -s

# 查看命令帮助
uv run fashion-search --help
```

## 运行数据

以下文件都生成在 `state/`，不会提交到 Git：

```text
state/
├── qdrant/                         # 双向量索引
├── index-report.json               # 建索引结果
├── same-product-calibrator.joblib  # 同款概率校准器
└── evaluation.json                 # 自动评测报告
```

## 当前规模与百万级扩展

2026-07-17 在本机实测扫描到 2 个店铺、6,799 个商品、37,874 张展示图，扫描错误为 0。这个规模仍可以先用 Qdrant Local 验证效果，但首次全量编码会明显比早期样本集耗时更长。以后达到百万商品时，核心检索流程不需要推翻，但建议：

- 每个商品挑选 3–5 张代表图；
- Qdrant Local 换成独立 Qdrant 服务或集群；
- 开启向量量化；
- 把离线建索引与在线查询拆开；
- 用消息队列增量处理新增商品；
- 增加服装、鞋、包、饰品的大类过滤；
- 将线性 pHash 扫描替换为专用二进制指纹索引。
