from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import joblib

from features import extract_image_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Загрузка обученной модели и предсказание класса по фото."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/plant_insect_classifier.joblib"),
        help="Файл .joblib, сохранённый train.py",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Путь к изображению для классификации",
    )
    return parser.parse_args()


def load_artifact(model_path: Path) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Не найден файл модели {model_path}. Сначала запустите train.py."
        )
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError(
            "Неверный формат артефакта. Обучите модель заново train.py."
        )
    return artifact


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Файл изображения {args.image} не найден.")

    artifact = load_artifact(args.model_path)
    pipeline = artifact["pipeline"]
    bins = int(artifact.get("bins_per_channel", 16))

    feature_vector = extract_image_features(args.image, bins_per_channel=bins)
    feature_vector = feature_vector.reshape(1, -1)

    predicted_class = pipeline.predict(feature_vector)[0]
    probabilities = pipeline.predict_proba(feature_vector)[0]
    classes = pipeline.classes_

    print(f"Предсказанный класс: {predicted_class}")
    print("\nВероятности по классам:")
    for label, prob in sorted(
        zip(classes, probabilities), key=lambda item: item[1], reverse=True
    ):
        print(f"  {label:10s}: {prob:.3f}")


if __name__ == "__main__":
    main()
