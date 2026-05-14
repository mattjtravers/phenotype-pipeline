from __future__ import annotations

from phenotype_pipeline.models import CleanSnpDataset, FeatureMatrix


def build_feature_matrix(
    dataset: CleanSnpDataset,
    maf_threshold: float = 0.01,
    association_filter: bool = False,
    top_n: int = 10_000,
) -> FeatureMatrix:
    """Applies marker selection filters (MAF, variance, optional association) and
    assembles the model-ready FeatureMatrix.

    Missingness filtering is owned by preprocessing — see preprocessing.preprocess().
    All filter statistics here are computed on training-split samples only.
    The resulting FeatureRegistry is applied unchanged to the test split.
    """
    raise NotImplementedError
