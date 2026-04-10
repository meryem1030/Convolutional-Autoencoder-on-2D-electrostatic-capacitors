#!/usr/bin/env python3
"""Train a convolutional autoencoder for 2D electrostatic capacitor contamination maps.

This script is a runnable/refactored version of the original notebook-exported code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D
from keras.models import Model
from sklearn.model_selection import train_test_split


def _safe_normalize(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize an array to [0, 1] safely."""
    denom = float(np.max(arr))
    if denom <= eps:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr / denom).astype(np.float64)


def load_grayscale_image(image_path: Path, size: Tuple[int, int]) -> np.ndarray:
    """Load a grayscale image, resize, and return shape (H, W, 1)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.resize(img, size)
    img = _safe_normalize(img)
    return img.reshape(size[1], size[0], 1)


def load_image_stack(images_dir: Path, size: Tuple[int, int]) -> np.ndarray:
    """Load all images in a directory into shape (N, H, W, 1)."""
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    image_paths = sorted([p for p in images_dir.iterdir() if p.is_file()])
    if not image_paths:
        raise ValueError(f"No image files found in: {images_dir}")

    images = [load_grayscale_image(p, size) for p in image_paths]
    return np.stack(images, axis=0).astype(np.float64)


def read_float_values(file_path: Path) -> List[float]:
    """Read whitespace-delimited float values from a text file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    values: List[float] = []
    with file_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            values.extend(float(item) for item in line.split())
    return values


def build_geometry_masks(
    sample_count: int,
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    *,
    A: float,
    B: float,
    W: float,
    H: float,
    Pc: float,
    size: Tuple[int, int],
) -> np.ndarray:
    """Create geometry-difference masks used as model inputs."""
    width_px, height_px = size
    if len(x_vals) < sample_count or len(y_vals) < sample_count:
        raise ValueError("axis/ordinate coordinates are fewer than image samples")

    wind = max(round(W / A * (width_px - 1)), 1)
    hind = max(round(H / B * (height_px - 1)), 1)
    d_p_ind = (Pc - 1) / 10

    x_idx = [max(round(x / A * (width_px - 1)), 1) for x in x_vals[:sample_count]]
    y_idx = [max(round(y / B * (height_px - 1)), 1) for y in y_vals[:sample_count]]

    masks = np.zeros((sample_count, height_px, width_px, 1), dtype=np.float64)

    for s in range(sample_count):
        x0 = min(x_idx[s], width_px - 1)
        y0 = min(y_idx[s], height_px - 1)
        x1 = min(x0 + wind, width_px)
        y1 = min(y0 + hind, height_px)
        masks[s, y0:y1, x0:x1, 0] = d_p_ind

    return masks


def build_autoencoder(height: int, width: int, in_channels: int = 1) -> Model:
    """Construct and compile convolutional autoencoder."""
    input_images = Input(shape=(height, width, in_channels))

    # Encoder
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(input_images)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    encoded = MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    model = Model(input_images, decoded)
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def prepare_targets(exact_images: np.ndarray, simplified: np.ndarray, threshold_ratio: float = 0.25) -> np.ndarray:
    """Build binary target maps from exact-simplified differences."""
    dif = np.abs(exact_images - simplified[np.newaxis, ...])
    thresholds = np.max(dif, axis=(1, 2, 3), keepdims=True) * threshold_ratio
    target = np.where(dif >= thresholds, 1.0, 0.0)
    return target.astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CAE on electrostatic capacitor data")
    parser.add_argument("--simplified-image", type=Path, required=True, help="Path to simplify.png")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory with exact/contaminated images")
    parser.add_argument("--axis-file", type=Path, required=True, help="Path to axisn.txt")
    parser.add_argument("--ordinate-file", type=Path, required=True, help="Path to ordinaten.txt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--A", type=float, default=6.0)
    parser.add_argument("--B", type=float, default=6.0)
    parser.add_argument("--W", type=float, default=0.2)
    parser.add_argument("--H", type=float, default=0.2)
    parser.add_argument("--Pc", type=float, default=8.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    size = (args.width, args.height)

    simplified = load_grayscale_image(args.simplified_image, size)
    exact_images = load_image_stack(args.images_dir, size)
    targets = prepare_targets(exact_images, simplified)

    x_vals = read_float_values(args.axis_file)
    y_vals = read_float_values(args.ordinate_file)
    inputs = build_geometry_masks(
        sample_count=len(exact_images),
        x_vals=x_vals,
        y_vals=y_vals,
        A=args.A,
        B=args.B,
        W=args.W,
        H=args.H,
        Pc=args.Pc,
        size=size,
    )

    train_E, valid_E, train_T, valid_T = train_test_split(
        inputs,
        targets,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = build_autoencoder(args.height, args.width, in_channels=1)
    history = model.fit(
        train_E,
        train_T,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        validation_data=(valid_E, valid_T),
    )

    print("Training complete")
    print(f"Final loss: {history.history['loss'][-1]:.6f}")
    print(f"Final val_loss: {history.history['val_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
