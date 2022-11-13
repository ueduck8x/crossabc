#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

CrossABCHistArray = Dict[Tuple[int, int], Dict[int, int]]
CrossABCDistArray = Dict[Tuple[int, int], Dict[int, float]]


class CrossABC:
    num_ranks = 10
    DEFAULT = 0
    IND1 = 1
    IND2 = 2

    def _check_input(self, df: pd.DataFrame, indicators: List[str]) -> None:
        if len(indicators) != 2:
            raise Exception("Length of indicators must be 2.")

        for i in indicators:
            if not (i in df.columns):
                raise KeyError(f"Indicator {i} does not exist in the df.columns.")

        # When df[i] does not be int or float
        for i in indicators:
            # The exception data portion is returned in NaN.
            # The other lines are converted to numerical values.
            df[i] = pd.to_numeric(df[i], errors="coerce")
            if not (df[i].notnull().all()):
                raise TypeError(f"There is a non-numeric element in df[{i}].")

        # When df[i] have negative value
        for i in indicators:
            if len(df[df[i] >= 0]) != len(df[i]):
                raise ValueError(f"Column must NOT contain negative values, df[{i}] contains negative values")

    def __init__(self, df: pd.DataFrame, indicators: List[str]) -> None:
        self._check_input(df, indicators)
        self._df = df
        inds = {i: name for i, name in enumerate(indicators, self.IND1)}
        self._indicators = inds
        axis_names = [f"rank_{i}" for i in range(1, self.num_ranks + 1)]
        self.axis_names = axis_names
        self._give_rank()
        self._histgrams: CrossABCHistArray = {}
        self._cum_histgrams: CrossABCHistArray = {}
        self._cum_distributions: CrossABCDistArray = {}
        self._make_histgrams()
        self._make_cum_histgrams()
        self._make_cum_distributions()

    def _give_rank(self) -> None:
        for n in self._indicators.values():
            self._df = self._df.sort_values(n, ascending=False)
            cum_ratio_series = self._df[n].cumsum() / self._df[n].sum()
            rank_series: pd.Series = np.ceil(cum_ratio_series * self.num_ranks)
            rank_series = rank_series.astype(int)
            rank_series.name = f"rank_{n}"
            self._df = pd.concat([self._df, rank_series], axis=1)

    # make histgram/cumrative distribution
    def _make_histgrams(self) -> None:
        # Summarize the elements with rank i for indicator 1 and rank j for indicator 2.
        groups = self._df.groupby([f"rank_{n}" for n in self._indicators.values()])
        for i in range(1, self.num_ranks + 1):
            for j in range(1, self.num_ranks + 1):
                cnt, ind1_sum, ind2_sum = 0, 0, 0
                try:
                    group_ij: pd.DataFrame = groups.get_group((i, j))
                    cnt = len(group_ij)
                    ind1_sum = group_ij[self._indicators[self.IND1]].sum()
                    ind2_sum = group_ij[self._indicators[self.IND2]].sum()
                except KeyError:
                    pass
                summary_group_ij = {self.DEFAULT: cnt, self.IND1: ind1_sum, self.IND2: ind2_sum}
                self._histgrams[i, j] = summary_group_ij

    def _make_cum_histgrams(self) -> None:
        self._cum_histgrams = copy.deepcopy(self._histgrams)
        for c in range(self.DEFAULT, self.IND2 + 1):
            # vertical zeta transform
            for i in range(2, self.num_ranks + 1):
                for j in range(1, self.num_ranks + 1):
                    self._cum_histgrams[i, j][c] += self._cum_histgrams[i - 1, j][c]
            # horizontal zeta transform
            for i in range(1, self.num_ranks + 1):
                for j in range(2, self.num_ranks + 1):
                    self._cum_histgrams[i, j][c] += self._cum_histgrams[i, j - 1][c]

    def _make_cum_distributions(self) -> None:
        for i in range(1, self.num_ranks + 1):
            for j in range(1, self.num_ranks + 1):
                self._cum_distributions[i, j] = {}
                for c in range(self.DEFAULT, self.IND2 + 1):
                    m = len(self._df) if c == self.DEFAULT else self._df[self._indicators[c]].sum()
                    v = self._cum_histgrams[i, j][c] / m
                    self._cum_distributions[i, j][c] = v

    # extract elements
    def _check_get_elements_input(self, ranks: List[int]) -> None:
        for r in ranks:
            if not (isinstance(r, int)):
                raise TypeError(f"{r} is str. The element of ranks must be an 'int'")
        for r in ranks:
            if r <= 0 or self.num_ranks < r:
                raise ValueError(f"{r} must be 0 < r < {self.num_ranks}")

    def get_elements(self, ranks: List[int]) -> pd.DataFrame:
        self._check_get_elements_input(ranks)
        rank_ind1, rank_ind2 = ranks[0], ranks[1]
        ind1_df = self._df[self._df[f"rank_{self._indicators[self.IND1]}"] <= rank_ind1]
        ans_df = ind1_df[ind1_df[f"rank_{self._indicators[self.IND2]}"] <= rank_ind2]
        return ans_df

    def get_df(self) -> pd.DataFrame:
        return self._df

    # get histgram/cum distribution
    def _check_mode(self, mode: int) -> None:
        if mode < self.DEFAULT or self.IND2 < mode:
            raise ValueError("mode must be 0 or 1 or 2 (0: number of elements, 1: indicator 1, 2: indicator 2)")

    def get_histgram_df(self, mode: int = 0, cumtype: bool = False) -> pd.DataFrame:
        self._check_mode(mode)
        hist = [[0 for _ in range(self.num_ranks)] for _ in range(self.num_ranks)]
        for i in range(1, self.num_ranks + 1):
            for j in range(1, self.num_ranks + 1):
                v = self._cum_histgrams[i, j][mode] if cumtype else self._histgrams[i, j][mode]
                hist[i - 1][j - 1] = v
        histgram_df = pd.DataFrame(hist, index=self.axis_names, columns=self.axis_names)
        return histgram_df

    def get_cum_distribution_df(self, mode: int = 0) -> pd.DataFrame:
        self._check_mode(mode)
        dist = [[0.0 for _ in range(self.num_ranks)] for _ in range(self.num_ranks)]
        for i in range(1, self.num_ranks + 1):
            for j in range(1, self.num_ranks + 1):
                v = self._cum_distributions[i, j][mode]
                dist[i - 1][j - 1] = v
        cumdist_df = pd.DataFrame(dist, index=self.axis_names, columns=self.axis_names)
        return cumdist_df
