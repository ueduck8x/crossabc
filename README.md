# crossabc
[![PyPI](https://img.shields.io/pypi/v/crossabc)](https://pypi.org/project/crossabc/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/crossabc)](https://pypi.org/project/crossabc/)
[![license](https://img.shields.io/github/license/hrt0809/crossabc)](https://github.com/hrt0809/crossabc/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![test](https://github.com/hrt0809/crossabc/actions/workflows/test.yml/badge.svg)](https://github.com/hrt0809/crossabc/actions/workflows/test.yml)

Easily CrossABC (Pareto) analyzer.

## Tutorial
Click [here(Google Colaboratory)](https://colab.research.google.com/drive/1Tw-aOTaSmgdJdYyjLH3Nqt2F3k6-Bi9A?usp=sharing) for the CrossABC Tutorial. Note that the Tutorial is written in **Japanese**.

## Usage
```Python
from crossabc import CrossABC
import pandas as pd

df = pd.DataFrame(
    data={
        "sales": {"item_1": 30750, "item_2": 29000},
        "profit": {"item_1": 8900, "item_2": 3430}
        }
    )
c = CrossABC(df=df, indicators=["sales", "profit"])
ans_df = c.get_df()
```
When ```df``` is

| | sales | profit |
| :--- | ---: | ---: |
item_1 | 30750 | 8900 |
item_2 | 29000 | 3430 |

and use ```CrossABC(df, ["sales", "profit"])```, now ```ans_df``` is

| | sales | profit | rank_sales | rank_profit |
| :--- | ---: | ---: | ---: | ---: |
item_1 | 30750 | 8900 | 6 | 8 |
item_2 | 29000 | 3430 | 10 | 10 |

## Build
The source code is currently hosted on GitHub at: https://github.com/hrt0809/crossabc. Binary installers for the latest released version are available at the [PyPI](https://pypi.org/project/crossabc/).

```
pip install crossabc
```

## Dependencies
1. https://numpy.org
1. https://pandas.pydata.org
