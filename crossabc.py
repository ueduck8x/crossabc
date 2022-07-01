#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Literal

import numpy as np
import pandas as pd


class CrossABC:
    def __init__(self) -> None:
        self.__df: pd.DataFrame
        self.__indicators: list[str]
        self.__division_number: int
        pass

    @classmethod
    def create(cls, df: pd.DataFrame, indicators: list[str], division_number: int = 10) -> CrossABC:
        cross_abc = cls()

        if len(df.columns) < 2:
            raise Exception('DataFrameオブジェクトの列数が2列未満です')
        # [NOTE] dfの中身がfloatかintで，Null要素がないことのチェックが必要

        if len(indicators) != 2:
            raise Exception('indicatorsで指定するindicatorの数は2つでなければなりません')
        if len(set(indicators)) < 2:
            raise Exception('indicatorsに同じ列名が入力されています')
        for indctr in indicators:
            if not (indctr in df.columns.values):
                raise Exception(f'第一引数に指定したdfの中にindicatorsで指定した"{indctr}"がありません．')
        if not isinstance(division_number, int):
            raise Exception('division_numberは整数型で指定する必要があります')

        cross_abc.__df = df
        cross_abc.__indicators = indicators
        cross_abc.__division_number = division_number
        cross_abc.__calculate_tier()

        return cross_abc

    # dfに格納されている各データにtierを付与する．
    def __calculate_tier(self) -> None:
        for i in self.__indicators:
            self.__df = self.__df.sort_values(i, ascending=False)
            # 現在のデータが上位何%に位置するデータなのか，各indicator，各データに対して計算する
            cum_ratio_df = self.__df[i].cumsum() / self.__df[i].sum()
            self.__df[f'cum_ratio_{i}'] = cum_ratio_df
            # 各データにtierを付与する
            # ex: あるデータについてcum_ratioが0.225(上位22.5%)のとき，tierはint(0.225 * 10) = 2
            tier_df = np.ceil(cum_ratio_df * self.__division_number).astype(int)
            self.__df[f'tier_{i}'] = tier_df

    def get_df(self) -> pd.DataFrame:
        return self.__df

    def get_histgram(self) -> pd.DataFrame:
        # tier_indicator_1とtier_indicator_2ブロックの中に属するデータ数を求める
        num_belong_tier_each_indicators = self.__df.groupby([f'tier_{i}' for i in self.__indicators]).size()

        # num_belong_tier_each_indicatorsをリストへ変換
        tier = int(100 / self.__division_number)
        histgram = [[0 for _ in range(tier, 100 + tier, tier)] for _ in range(tier, 100 + tier, tier)]
        for key, value in num_belong_tier_each_indicators.to_dict().items():
            tier_indicator_1, tier_indicator_2 = key
            histgram[tier_indicator_1 - 1][tier_indicator_2 - 1] = value

        # pd.DataFrameへと変換
        axis_name: list[str] = [f'{p}%' for p in range(tier, 100 + 1, tier)]
        histgram_df = pd.DataFrame(histgram, index=axis_name, columns=axis_name)

        # 最初に指定したindicatorが横軸にくるよう転置
        histgram_df = histgram_df.T
        return histgram_df

    def get_elements(self, ratios: list[tuple[float, Literal['>', '>=', '<', '<=']]]) -> pd.DataFrame:
        if len(ratios) != 2:
            raise Exception('ratiosの要素数は2つでなければなりません')

        ans_abc = CrossABC.create(self.__df, self.__indicators)
        ans_df = ans_abc.get_df()

        # ratiosで指定された条件で抽出作業をおこなう
        for i, tp in zip(self.__indicators, ratios):
            ratio, operator = tp
            if operator == '>':
                ans_df = ans_df[ans_df[f'cum_ratio_{i}'] > ratio]
            elif operator == '>=':
                ans_df = ans_df[ans_df[f'cum_ratio_{i}'] >= ratio]
            elif operator == '<':
                ans_df = ans_df[ans_df[f'cum_ratio_{i}'] < ratio]
            else:
                ans_df = ans_df[ans_df[f'cum_ratio_{i}'] <= ratio]

        # 内部の計算に用いた不要なデータは削除してから返す
        for i in self.__indicators:
            ans_df = ans_df.drop(columns=[f'tier_{i}', f'cum_ratio_{i}'])

        return ans_df
