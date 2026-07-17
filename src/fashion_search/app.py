"""仅监听本机的 Gradio 搜索页面。"""

from __future__ import annotations

from pathlib import Path

import gradio as gr


def _gallery(results):
    items = []
    for item in results:
        probability = (
            ""
            if item.same_probability is None
            else f" · 同款概率 {item.same_probability:.3f}"
        )
        caption = (
            f"店铺 {item.shop_id} · 商品 {item.product_id} · "
            f"分数 {item.score:.3f}{probability}"
        )
        items.append((str(item.representative_path), caption))
    return items


def build_app(search_service=None, *, status_message: str = "准备就绪"):
    with gr.Blocks(title="服饰以图搜商品") as app:
        gr.Markdown(
            "# 服饰以图搜商品\n上传服装、鞋、包或饰品图片，优先展示高置信同款，再展示相似款。"
        )
        status = gr.Markdown(status_message)
        with gr.Row():
            query = gr.Image(type="filepath", label="查询图片")
            weight = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.8,
                step=0.05,
                label="FashionSigLIP 权重（越高越重视品类、颜色、材质和风格）",
            )
        button = gr.Button("搜索", variant="primary")
        gr.Markdown("## 可能同款")
        same_gallery = gr.Gallery(label="高置信同款", columns=5, height="auto")
        gr.Markdown("## 相似商品")
        similar_gallery = gr.Gallery(label="相似款", columns=5, height="auto")

        def search(image_path, fashion_weight):
            if search_service is None:
                return [], [], "尚未准备好：请先执行 `uv run fashion-search index` 和 `calibrate`。"
            if not image_path:
                return [], [], "请先上传一张图片。"
            try:
                response = search_service.search(
                    Path(image_path), fashion_weight=float(fashion_weight)
                )
                message = (
                    f"找到 {len(response.same_products)} 个可能同款，"
                    f"{len(response.similar_products)} 个相似商品。"
                )
                return _gallery(response.same_products), _gallery(response.similar_products), message
            except Exception as exc:
                return [], [], f"搜索失败：{exc}"

        button.click(
            search,
            inputs=[query, weight],
            outputs=[same_gallery, similar_gallery, status],
        )
    return app
