#!/usr/bin/env python3
"""Create oracle condition-wise mean-aligned PLISM embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


NUM_COORD_COLUMNS = 3
EXPECTED_NUM_SLIDES = 91


def parse_condition(slide_name: str) -> tuple[str, str]:
    """Extract staining and scanner from a PLISM slide directory name."""
    parts = slide_name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse PLISM condition from {slide_name!r}.")
    return parts[0], parts[1]


def find_feature_paths(
    features_root: Path,
    extractor: str,
) -> list[Path]:
    """Find the 91 registered PLISM feature files."""
    extractor_dir = features_root / extractor

    if not extractor_dir.is_dir():
        raise FileNotFoundError(f"Missing extractor directory: {extractor_dir}")

    feature_paths = sorted(
        path
        for path in extractor_dir.glob("*/features.npy")
        if "_to_GMH_S60" in path.parent.name
    )

    if len(feature_paths) != EXPECTED_NUM_SLIDES:
        raise RuntimeError(
            f"Expected {EXPECTED_NUM_SLIDES} feature files, "
            f"found {len(feature_paths)} in {extractor_dir}."
        )

    return feature_paths


def find_reference_path(
    feature_paths: list[Path],
    reference_staining: str,
    reference_scanner: str,
) -> Path:
    """Find the feature file corresponding to the reference condition."""
    matches = [
        path
        for path in feature_paths
        if parse_condition(path.parent.name) == (reference_staining, reference_scanner)
    ]

    if len(matches) != 1:
        names = [path.parent.name for path in matches]
        raise RuntimeError(
            "Expected exactly one reference condition "
            f"{reference_staining}_{reference_scanner}, found {names}."
        )

    return matches[0]


def embedding_mean(feature_path: Path) -> np.ndarray:
    """Compute the slide embedding mean with float64 accumulation."""
    matrix = np.load(feature_path, mmap_mode="r")

    if matrix.ndim != 2 or matrix.shape[1] <= NUM_COORD_COLUMNS:
        raise ValueError(f"Invalid feature matrix {feature_path}: shape={matrix.shape}")

    return matrix[:, NUM_COORD_COLUMNS:].mean(
        axis=0,
        dtype=np.float64,
    )


def align_features(
    *,
    features_root: Path,
    extractor: str,
    output_extractor: str,
    reference_staining: str = "GMH",
    reference_scanner: str = "S60",
    overwrite: bool = False,
) -> None:
    """Align every PLISM condition mean to the reference-condition mean."""
    if extractor == output_extractor:
        raise ValueError("Input and output extractor names must differ.")

    feature_paths = find_feature_paths(features_root, extractor)
    reference_path = find_reference_path(
        feature_paths,
        reference_staining,
        reference_scanner,
    )

    reference_matrix = np.load(reference_path, mmap_mode="r")
    reference_coordinates = np.asarray(
        reference_matrix[:, :NUM_COORD_COLUMNS],
        dtype=np.float32,
    )
    reference_mean = embedding_mean(reference_path)

    output_root = features_root / output_extractor
    output_root.mkdir(parents=True, exist_ok=True)

    slide_names: list[str] = []
    slide_means: list[np.ndarray] = []
    slide_shifts: list[np.ndarray] = []

    print(f"Reference slide: {reference_path.parent.name}")
    print(f"Embedding dimension: {reference_mean.shape[0]}")

    for feature_path in feature_paths:
        slide_name = feature_path.parent.name
        output_path = output_root / slide_name / "features.npy"

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Use --overwrite to replace it."
            )

        matrix = np.load(feature_path, mmap_mode="r")

        np.testing.assert_allclose(
            matrix[:, :NUM_COORD_COLUMNS],
            reference_coordinates,
            err_msg=f"Coordinate mismatch for {slide_name}",
        )

        condition_mean = matrix[:, NUM_COORD_COLUMNS:].mean(
            axis=0,
            dtype=np.float64,
        )

        # Estimated oracle acquisition displacement relative to GMH/S60.
        condition_shift = condition_mean - reference_mean

        # x_aligned = x - mu_condition + mu_reference
        aligned_embeddings = np.asarray(
            matrix[:, NUM_COORD_COLUMNS:],
            dtype=np.float32,
        ) - condition_shift.astype(np.float32)

        output = np.empty(matrix.shape, dtype=np.float32)
        output[:, :NUM_COORD_COLUMNS] = matrix[:, :NUM_COORD_COLUMNS]
        output[:, NUM_COORD_COLUMNS:] = aligned_embeddings

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, output)

        # Sanity check: each corrected slide mean should equal the reference mean.
        aligned_mean = aligned_embeddings.mean(axis=0, dtype=np.float64)
        max_mean_error = float(np.max(np.abs(aligned_mean - reference_mean)))

        print(
            f"{slide_name}: "
            f"shift_norm={np.linalg.norm(condition_shift):.4f}, "
            f"max_mean_error={max_mean_error:.3e}"
        )

        slide_names.append(slide_name)
        slide_means.append(condition_mean)
        slide_shifts.append(condition_shift)

    np.savez_compressed(
        output_root / "oracle_mean_alignment.npz",
        slide_names=np.asarray(slide_names),
        slide_means=np.stack(slide_means),
        slide_shifts=np.stack(slide_shifts),
        reference_mean=reference_mean,
        reference_slide=np.asarray(reference_path.parent.name),
    )

    metadata = {
        "method": "oracle_condition_specific_mean_alignment",
        "input_extractor": extractor,
        "output_extractor": output_extractor,
        "reference_staining": reference_staining,
        "reference_scanner": reference_scanner,
        "reference_slide": reference_path.parent.name,
        "num_slides": len(feature_paths),
        "formula": "x_aligned = x - mean_condition + mean_reference",
        "uses_evaluation_condition_statistics": True,
    }

    with (output_root / "oracle_mean_alignment.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)

    print(f"Aligned features written to: {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--output-extractor", required=True)
    parser.add_argument("--reference-staining", default="GMH")
    parser.add_argument("--reference-scanner", default="S60")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    align_features(
        features_root=args.features_root,
        extractor=args.extractor,
        output_extractor=args.output_extractor,
        reference_staining=args.reference_staining,
        reference_scanner=args.reference_scanner,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
