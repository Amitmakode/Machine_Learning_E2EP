import pandas as pd
import pytest

from laptop_price.components.data_validation import validate
from laptop_price.exception import PricePredictorException


def test_validate_accepts_valid_dataframe():
    assert validate(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), ["a", "b"])


def test_validate_rejects_missing_columns():
    assert not validate(pd.DataFrame({"a": [1]}), ["a", "b"])


def test_validate_rejects_excessive_nulls():
    assert not validate(pd.DataFrame({"a": [None, None, 1]}), ["a"], na_threshold=0.5)


def test_validate_rejects_empty_dataframe():
    assert not validate(pd.DataFrame(), [])


def test_validate_rejects_invalid_threshold():
    with pytest.raises(PricePredictorException, match="na_threshold"):
        validate(pd.DataFrame({"a": [1]}), ["a"], na_threshold=2)
