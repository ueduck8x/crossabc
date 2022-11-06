#####! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from inspect import signature

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

CrossABCHistArray = dict[tuple[int, int], dict[int, int]]
CrossABCDistArray = dict[tuple[int, int], dict[int, float]]


class CrossABC:
    num_ranks = 10
    DEFAULT = 0
    IND1 = 1
    IND2 = 2
    STAR = "star"
    PROB_CHILD = "prob_child"
    CASH_COW = "cash_cow"
    DOG = "dog"

    def __init__(self, df: pd.DataFrame) -> None:
        self._check_input(df)
        self.axis_names = [f"rank_{i}" for i in range(1, self.num_ranks + 1)]
        inds = {i: name for i, name in enumerate(df.columns, self.IND1)}
        self._indicators = inds
        self._df = df
        self._histgrams: CrossABCHistArray = {}
        self._cum_histgrams: CrossABCHistArray = {}
        self._cum_distributions: CrossABCDistArray = {}
        self._give_rank()
        self._make_histgrams()
        self._make_cum_histgrams()
        self._make_cum_distributions()

    # 入力チェック
    def _check_input(self, df: pd.DataFrame) -> None:
        if len(df.columns) != 2:
            raise Exception("Number of elements in df must be 2!")

    # ランク付け作業
    def _give_rank(self) -> None:
        for n in self._indicators.values():
            self._df = self._df.sort_values(n, ascending=False)
            cum_ratio_series = self._df[n].cumsum() / self._df[n].sum()
            rank_series = np.ceil(cum_ratio_series * self.num_ranks).astype(int)
            rank_series.name = f"rank_{n}"
            self._df = pd.concat([self._df, rank_series], axis=1)

    # ヒストグラム/累積分布 作成
    def _make_histgrams(self) -> None:
        rank_group = self._df.groupby([f"rank_{n}" for n in self._indicators.values()])
        for i in range(1, self.num_ranks + 1):
            for j in range(1, self.num_ranks + 1):
                cnt, ind1_sum, ind2_sum = 0, 0, 0
                try:
                    rank_i_j_df: pd.DataFrame = rank_group.get_group((i, j))
                    cnt = len(rank_i_j_df)
                    ind1_sum = rank_i_j_df[self._indicators[self.IND1]].sum()
                    ind2_sum = rank_i_j_df[self._indicators[self.IND2]].sum()
                except KeyError:
                    pass
                rank_i_j = {self.DEFAULT: cnt, self.IND1: ind1_sum, self.IND2: ind2_sum}
                self._histgrams[i, j] = rank_i_j

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

    # 要素抽出
    def get_elements(self, *, ppm: str | None = None, threshold: float | None = None, mode: int = 0) -> pd.DataFrame:
        if ppm is None and threshold is None:
            raise Exception("Set the value to ONE of ppm and threshold")
        if (ppm is not None) and (not isinstance(ppm, str)):
            raise TypeError()
        if (threshold is not None) and (not isinstance(threshold, float)):
            raise TypeError()
        if isinstance(threshold, float) and isinstance(ppm, str):
            raise Exception(f"Set either {ppm = } or {threshold = } to None")
        ans_df = pd.DataFrame()
        if isinstance(threshold, float):
            ans_df = self._get_elements_threshold(threshold, mode)
        elif isinstance(ppm, str):
            ans_df = self._get_elements_ppm(ppm)
        for n in self._indicators.values():
            ans_df = ans_df.sort_values(f"rank_{n}")
        return ans_df

    def _get_elements_threshold(self, threshold: float, mode: int) -> pd.DataFrame:
        cum_dfs = []
        for i in range(1, self.num_ranks + 1):
            for j in range(1, self.num_ranks + 1):
                v = self._cum_distributions[i, j][mode]
                if v < threshold:
                    ind1_df = self._df[self._df[f"rank_{self._indicators[self.IND1]}"] == i]
                    ind2_df = ind1_df[ind1_df[f"rank_{self._indicators[self.IND2]}"] == j]
                    cum_dfs.append(ind2_df)
        cum_df = pd.concat(cum_dfs)
        cum_df.drop_duplicates(inplace=True)
        return cum_df

    def _get_elements_ppm(self, ppm: str) -> pd.DataFrame:
        ppm_cats = [self.STAR, self.PROB_CHILD, self.CASH_COW, self.DOG]
        if not (ppm in ppm_cats):
            raise Exception(f"Select one value from {ppm_cats} and input it into ppm")
        mid_thrshld = int(self.num_ranks // 2)
        ind1_df = pd.DataFrame()
        if ppm in [self.STAR, self.PROB_CHILD]:
            ind1_df = self._df[self._df[f"rank_{self._indicators[self.IND1]}"] <= mid_thrshld]
        else:
            ind1_df = self._df[self._df[f"rank_{self._indicators[self.IND1]}"] > mid_thrshld]
        ind2_df = pd.DataFrame()
        if ppm in [self.STAR, self.CASH_COW]:
            ind2_df = ind1_df[ind1_df[f"rank_{self._indicators[self.IND2]}"] <= mid_thrshld]
        else:
            ind2_df = ind1_df[ind1_df[f"rank_{self._indicators[self.IND2]}"] > mid_thrshld]
        return ind2_df

    def get_df(self) -> pd.DataFrame:
        return self._df

    # 描画
    def draw_histgram(  # type: ignore
        self, ax: Axes, mode: int = 0, cumtype: bool = False, threshold: float | None = None, **kwargs
    ) -> None:
        self._check_kwds(kwargs)
        df = self.get_histgram(mode, cumtype)
        self._draw(ax, df, mode, threshold, fmt="d", **kwargs)

    def draw_cum_distribution(  # type: ignore
        self, ax: Axes, mode: int = 0, threshold: float | None = None, **kwargs
    ) -> None:
        self._check_kwds(kwargs)
        df = self.get_cum_distribution(mode)
        self._draw(ax, df, mode, threshold, fmt=".4f", **kwargs)

    def _check_kwds(self, kwags) -> None:  # type: ignore
        valid_kwds = signature(self._draw).parameters.keys()
        if any([k not in valid_kwds for k in kwags]):
            invalid_args = ", ".join([k for k in kwags if k not in valid_kwds])
            raise ValueError(f"Received invalid argument(s): {invalid_args}")

    def get_histgram(self, mode: int = 0, cumtype: bool = False) -> pd.DataFrame:
        hist = [[0 for _ in range(self.num_ranks)] for _ in range(self.num_ranks)]
        for i in range(1, self.num_ranks + 1):
            for j in range(1, self.num_ranks + 1):
                v = self._cum_histgrams[i, j][mode] if cumtype else self._histgrams[i, j][mode]
                hist[i - 1][j - 1] = v
        df = pd.DataFrame(hist, index=self.axis_names, columns=self.axis_names)
        return df

    def get_cum_distribution(self, mode: int = 0) -> pd.DataFrame:
        dist = [[0.0 for _ in range(self.num_ranks)] for _ in range(self.num_ranks)]
        for i in range(1, self.num_ranks + 1):
            for j in range(1, self.num_ranks + 1):
                v = self._cum_distributions[i, j][mode]
                dist[i - 1][j - 1] = v
        df = pd.DataFrame(dist, index=self.axis_names, columns=self.axis_names)
        return df

    def _draw(
        self,
        ax: Axes,
        df: pd.DataFrame,
        mode: int,
        threshold: float | None,
        fmt: str,
        *,
        fontsize: int = 15,
        label_fontsize: int = 15,
        linewidth: int = 2,
        linecolor: str = "black",
        cmap: str = "Spectral_r",
    ) -> None:
        sns.heatmap(df, annot=True, fmt=fmt, annot_kws={"fontsize": fontsize}, cmap=cmap, ax=ax)
        ax.set_ylabel(self._indicators[self.IND1], fontsize=label_fontsize)
        ax.set_xlabel(self._indicators[self.IND2], fontsize=label_fontsize)
        if threshold is not None:
            self._draw_line(ax, mode, threshold, linewidth, linecolor)

    def _draw_line(self, ax: Axes, mode: int, threshold: float, linewidth: int, linecolor: str) -> None:
        for i in range(1, self.num_ranks + 1):
            row_break, col_break = False, False
            for j in reversed(range(1, self.num_ranks + 1)):
                v_row = self._cum_distributions[i, j][mode]
                v_col = self._cum_distributions[j, i][mode]
                if v_row < threshold and row_break is False:
                    ax.axvline(
                        x=j - 0.01, ymin=1 - 0.1 * i, ymax=1 - 0.1 * (i - 1), linewidth=linewidth, color=linecolor
                    )
                    row_break = True
                if v_col < threshold and col_break is False:
                    ax.axhline(y=j - 0.01, xmin=0.1 * (i - 1), xmax=0.1 * i, linewidth=linewidth, color=linecolor)
                    col_break = True
