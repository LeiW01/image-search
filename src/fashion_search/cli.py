"""项目命令行：检查、索引、校准、评测与本地页面。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .calibration import SameProductCalibrator, build_training_pairs
from .config import Settings
from .encoders import DinoV2Encoder, FashionSiglipEncoder
from .evaluate import evaluate_dataset
from .indexer import Indexer
from .scanner import scan_display_images
from .search import SearchService
from .vector_store import VectorStore


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fashion-search", description="本地服饰商品双模型以图搜商品"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="检查商品展示图目录，不加载模型")
    index = commands.add_parser("index", help="下载模型并增量建立双向量索引")
    index.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只索引前 N 张展示图用于小规模试跑；省略则全量索引",
    )
    commands.add_parser("calibrate", help="用商品 ID 校准高置信同款概率")
    evaluate = commands.add_parser("evaluate", help="自动评测双模型与融合权重")
    evaluate.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每种配置最多评测 N 张已索引图片；省略则评测全部已索引图片",
    )
    commands.add_parser("serve", help="启动仅限本机访问的 Gradio 页面")
    return parser


def _settings() -> Settings:
    return Settings.default(project_root())


def _service(settings: Settings, store: VectorStore | None = None) -> SearchService:
    calibrator_path = settings.state_dir / "same-product-calibrator.joblib"
    if not calibrator_path.is_file():
        raise RuntimeError("尚未校准同款，请先运行：uv run fashion-search calibrate")
    return SearchService(
        settings,
        store or VectorStore(settings),
        FashionSiglipEncoder(settings),
        DinoV2Encoder(settings),
        SameProductCalibrator.load(calibrator_path),
    )


def command_inspect(settings: Settings) -> None:
    records, errors = scan_display_images(settings.image_root, compute_digest=False)
    shops = {record.shop_id for record in records}
    products = {record.product_key for record in records}
    print(f"图片目录：{settings.image_root}")
    print(f"店铺数：{len(shops)}")
    print(f"商品数：{len(products)}")
    print(f"展示图数：{len(records)}")
    print(f"扫描错误数：{len(errors)}")


def command_index(settings: Settings, *, limit: int | None = None) -> None:
    store = VectorStore(settings)
    try:
        stats = Indexer(
            settings,
            store,
            FashionSiglipEncoder(settings),
            DinoV2Encoder(settings),
        ).run(limit=limit)
        print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
    finally:
        store.close()


def command_calibrate(settings: Settings) -> None:
    store = VectorStore(settings)
    try:
        features, labels = build_training_pairs(store.all_points(with_vectors=True))
        if len(features) < 4 or len(np.unique(labels)) < 2:
            raise RuntimeError("同款训练样本不足；请确认索引中存在多个商品且部分商品有多张展示图")
        train_x, valid_x, train_y, valid_y = train_test_split(
            features, labels, test_size=0.25, random_state=42, stratify=labels
        )
        calibrator = SameProductCalibrator(target_precision=0.95).fit(
            train_x, train_y, valid_x, valid_y
        )
        output = settings.state_dir / "same-product-calibrator.joblib"
        calibrator.save(output)
        print(f"校准完成：样本 {len(features)}，同款阈值 {calibrator.threshold:.4f}")
        print(f"保存位置：{output}")
    finally:
        store.close()


def command_evaluate(settings: Settings, *, limit: int | None = None) -> None:
    store = VectorStore(settings)
    try:
        records, errors = scan_display_images(settings.image_root, compute_digest=False)
        indexed_ids = set(store.indexed_payloads())
        records = [record for record in records if record.image_id in indexed_ids]
        if errors:
            print(f"扫描时记录到 {len(errors)} 个错误，将跳过这些文件。")
        report = evaluate_dataset(
            _service(settings, store),
            records,
            settings.state_dir / "evaluation.json",
            query_limit=limit,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        store.close()


def command_serve(settings: Settings) -> None:
    from .app import build_app

    store = VectorStore(settings)
    if not store.client.collection_exists(settings.collection_name):
        store.close()
        raise RuntimeError("尚未建立索引，请先运行：uv run fashion-search index")
    try:
        app = build_app(_service(settings, store))
        app.launch(server_name=settings.bind_host, server_port=settings.bind_port, share=False)
    finally:
        store.close()


def main() -> None:
    args = build_parser().parse_args()
    settings = _settings()
    try:
        if args.command == "index":
            command_index(settings, limit=args.limit)
        elif args.command == "evaluate":
            command_evaluate(settings, limit=args.limit)
        else:
            commands = {
                "inspect": command_inspect,
                "calibrate": command_calibrate,
                "serve": command_serve,
            }
            commands[args.command](settings)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}") from exc
