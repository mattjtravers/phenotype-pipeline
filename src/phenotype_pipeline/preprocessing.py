from __future__ import annotations

from phenotype_pipeline.models import CleanSnpDataset, RawSnpDataset


class PreprocessingError(Exception):
    pass


def preprocess(
    dataset: RawSnpDataset,
    random_state: int = 42,
) -> CleanSnpDataset:
    """Splits, imputes, encodes, and validates the raw dataset.

    Returns a CleanSnpDataset with per-sample split assignments, dosage encoding,
    and imputation medians computed exclusively from training-split samples.
    """
    raise NotImplementedError
