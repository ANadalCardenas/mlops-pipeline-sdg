import numpy as np
from sklearn.pipeline import Pipeline

from src.features.build_features import EXCLUDED_COLUMNS, TARGET, build_training_pipeline


def test_pipeline_structure_excludes_target_and_excluded_columns(sdg_dataframe):
    pipeline = build_training_pipeline(sdg_dataframe)

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocessor", "classifier"]

    preprocessor = pipeline.named_steps["preprocessor"]
    used_columns = {col for _, _, cols in preprocessor.transformers for col in cols}
    assert used_columns == {"F1", "F5", "F22"}
    assert TARGET not in used_columns
    assert not used_columns & set(EXCLUDED_COLUMNS)


def test_pipeline_fits_and_predicts_with_missing_values(sdg_dataframe):
    df = sdg_dataframe.copy()
    df.loc[0, "F22"] = np.nan
    df.loc[1, "F1"] = np.nan

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    pipeline = build_training_pipeline(X)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(df)
    assert set(preds).issubset({"A", "B", "C", "D"})
