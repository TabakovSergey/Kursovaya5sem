from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import extract_image_features, iter_image_paths

DATASET_GROUPS: Dict[str, List[str]] = {
    "insects": ["armyworm", "mosquito", "sawfly"],
    "plants": ["banana", "coconut", "cucumber"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение простого классификатора растений и насекомых."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets"),
        help="Корневая директория с папками insects/ и plants/",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/plant_insect_classifier.joblib"),
        help="Куда сохранить обученный пайплайн",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Доля выборки для теста (0 < value < 1)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed для воспроизводимости train/test split",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=16,
        help="Количество бинов цветовой гистограммы на канал",
    )
    return parser.parse_args()


def collect_dataset(
    dataset_root: Path, bins_per_channel: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Читает изображения из подготовленных папок и возвращает признаки и метки.
    """
    features: List[np.ndarray] = []
    labels: List[str] = []

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Директория с датасетом '{dataset_root}' не найдена. "
            "Проверьте --data-root."
        )

    for group_name, class_names in DATASET_GROUPS.items():
        for class_name in class_names:
            class_dir = dataset_root / group_name / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Не найдена папка {class_dir}. "
                    "Соблюдайте структуру datasets/<группа>/<класс>/"
                )

            added = 0
            for img_path in iter_image_paths(class_dir):
                vector = extract_image_features(
                    img_path, bins_per_channel=bins_per_channel
                )
                features.append(vector)
                labels.append(class_name)
                added += 1

            if added == 0:
                print(f"[!] Внимание: в {class_dir} нет изображений.")

    if not features:
        raise RuntimeError("Не найдено ни одного изображения для обучения.")

    X = np.vstack(features)
    y = np.array(labels)
    return X, y


def build_model() -> Pipeline:
    """
    Возвращает sklearn-пайплайн: стандартизация -> логистическая регрессия.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    X, y = collect_dataset(args.data_root, bins_per_channel=args.bins)

    class_counter = Counter(y)
    print("Классы и количество изображений:")
    for class_name, count in class_counter.items():
        print(f"  {class_name:10s}: {count}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = build_model()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nТочность (accuracy) на тестовой выборке: {accuracy:.3f}\n")
    print("Отчёт классификации:\n")
    print(classification_report(y_test, y_pred))

    artifact = {
        "pipeline": pipeline,
        "bins_per_channel": args.bins,
        "class_names": pipeline.classes_,
    }
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_path)
    print(f"Модель сохранена в {args.model_path}")


if __name__ == "__main__":
    main()
