import numpy as np
import pandas as pd
import pytest

from crossabc.crossabc import CrossABC


def test_input_indicators() -> None:
    data = {"item_1": {"sales": 30750, "profit": 8900}, "item_2": {"sales": 29000, "profit": 3430}}
    false_indicators_1 = ["sales", "profit", "sales_figures"]
    false_indicators_2 = ["sales", "profiit"]
    indicators = ["sales", "profit"]
    df = pd.DataFrame(data).T
    # test 1: length not 2
    with pytest.raises(Exception) as e:
        _ = CrossABC(df, false_indicators_1)
    assert str(e.value) == "Length of indicators must be 2."
    # test 2: KeyError
    with pytest.raises(KeyError) as e:
        _ = CrossABC(df, false_indicators_2)
    assert str(e.value) == "'Indicator profiit does not exist in the df.columns.'"
    # test 3: normal
    CrossABC(df, indicators)


def test_input_dataframe_str() -> None:
    data = {"item_1": {"sales": "30750円", "profit": 8900}, "item_2": {"sales": 29000, "profit": 3430}}
    df = pd.DataFrame(data).T
    with pytest.raises(Exception) as e:
        _ = CrossABC(df, list(df.columns))
    assert str(e.value) == "There is a non-numeric element in df[sales]."


def test_input_dataframe_str_can_convert_int() -> None:
    data = {"item_1": {"sales": 30750, "profit": "8900"}, "item_2": {"sales": 29000, "profit": 3430}}
    df = pd.DataFrame(data).T
    CrossABC(df, list(df.columns))


def test_input_dataframe_nan() -> None:
    data = {"item_1": {"sales": 30750, "profit": 8900}, "item_2": {"sales": 29000, "profit": np.nan}}
    df = pd.DataFrame(data).T
    with pytest.raises(TypeError) as e:
        _ = CrossABC(df, list(df.columns))
    assert str(e.value) == "There is a non-numeric element in df[profit]."


def test_input_dataframe_negative_value() -> None:
    data = {"item_1": {"sales": -30750, "profit": 8900}, "item_2": {"sales": 29000, "profit": 3430}}
    df = pd.DataFrame(data).T
    with pytest.raises(ValueError) as e:
        _ = CrossABC(df, list(df.columns))
    assert str(e.value) == "Column must NOT contain negative values, df[sales] contains negative values"


def test_give_rank() -> None:
    data = {
        "item_1": {"sales": 30750, "profit": 8900},
        "item_2": {"sales": 29000, "profit": 3430},
        "item_3": {"sales": 28700, "profit": 6400},
        "item_4": {"sales": 68000, "profit": 47000},
        "item_5": {"sales": 45540, "profit": 13000},
        "item_6": {"sales": 39800, "profit": 22000},
        "item_7": {"sales": 27500, "profit": 2650},
        "item_8": {"sales": 42000, "profit": 16530},
        "item_9": {"sales": 42000, "profit": 25000},
        "item_10": {"sales": 28710, "profit": 6930},
        "item_11": {"sales": 118000, "profit": 90400},
    }
    rank_sales = [3, 4, 7, 8, 6, 5, 8, 9, 10, 9, 10]
    rank_profit = [4, 6, 7, 8, 9, 9, 10, 10, 10, 10, 10]

    df = pd.DataFrame(data).T
    cc = CrossABC(df, list(df.columns))
    ranked_df = cc.get_df()
    assert ranked_df["rank_sales"].tolist() == rank_sales
    assert ranked_df["rank_profit"].tolist() == rank_profit


def test_give_rank_2() -> None:
    data = {
        "item_1": {"sales": 100, "profit": 100},
        "item_2": {"sales": 100, "profit": 100},
        "item_3": {"sales": 100, "profit": 100},
        "item_4": {"sales": 100, "profit": 100},
        "item_5": {"sales": 100, "profit": 100},
        "item_6": {"sales": 100, "profit": 100},
        "item_7": {"sales": 100, "profit": 100},
        "item_8": {"sales": 100, "profit": 100},
        "item_9": {"sales": 100, "profit": 100},
        "item_10": {"sales": 100, "profit": 100},
    }
    rank_sales = [i for i in range(1, 10 + 1)]
    rank_profit = [i for i in range(1, 10 + 1)]

    df = pd.DataFrame(data).T
    cc = CrossABC(df, list(df.columns))
    ranked_df = cc.get_df()
    assert ranked_df["rank_sales"].tolist() == rank_sales
    assert ranked_df["rank_profit"].tolist() == rank_profit
