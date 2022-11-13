# crossabc
[![PyPI](https://img.shields.io/pypi/v/crossabc)](https://pypi.org/project/crossabc/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/crossabc)](https://pypi.org/project/crossabc/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license](https://img.shields.io/github/license/hrt0809/crossabc)](https://github.com/hrt0809/crossabc/blob/main/LICENSE)

Easily CrossABC analyzer

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
