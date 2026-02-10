"""
技术指标计算模块
实现MA/MACD/RSI/KDJ/布林带/威廉指标/DMI/CCI等技术指标
"""

import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from .signals import IndicatorSignal, SignalType

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """技术指标计算类"""

    @staticmethod
    def ma(data: pd.Series, periods: List[int] = None) -> Dict[int, float]:
        """
        计算移动平均线
        :param data: 价格数据
        :param periods: 周期列表
        :return: 各周期MA值字典
        """
        if periods is None:
            periods = [5, 10, 20, 60]

        ma_values = {}
        for period in periods:
            ma_values[period] = data.rolling(window=period).mean().iloc[-1]

        return ma_values

    @staticmethod
    def ema(data: pd.Series, periods: List[int] = None) -> Dict[int, float]:
        """
        计算指数移动平均线
        :param data: 价格数据
        :param periods: 周期列表
        :return: 各周期EMA值字典
        """
        if periods is None:
            periods = [5, 10, 20, 60]

        ema_values = {}
        for period in periods:
            ema_values[period] = data.ewm(span=period).mean().iloc[-1]

        return ema_values

    @staticmethod
    def macd(data: pd.Series,
             fast_period: int = 12,
             slow_period: int = 26,
             signal_period: int = 9) -> Dict[str, float]:
        """
        计算MACD指标
        :param data: 价格数据
        :param fast_period: 快速EMA周期
        :param slow_period: 慢速EMA周期
        :param signal_period: 信号EMA周期
        :return: MACD指标字典
        """
        # 计算快速EMA和慢速EMA
        ema_fast = data.ewm(span=fast_period).mean()
        ema_slow = data.ewm(span=slow_period).mean()

        # 计算MACD线 (DIF)
        dif = ema_fast - ema_slow

        # 计算信号线 (DEA)
        dea = dif.ewm(span=signal_period).mean()

        # 计算MACD柱 (MACD Histogram)
        macd_hist = 2 * (dif - dea)

        macd_value = macd_hist.iloc[-1]

        # 简单的信号方向：大于0视为多头，小于等于0视为空头
        macd_signal = 1.0 if macd_value > 0 else -1.0

        # 为保持向后兼容，提供 'macd' 和 'signal' 键
        return {
            'dif': dif.iloc[-1],
            'dea': dea.iloc[-1],
            'macd_hist': macd_value,
            'macd': macd_value,
            'signal': macd_signal,
            'histogram': macd_value,
        }

    @staticmethod
    def rsi(data: pd.Series, period: int = 14,
            oversold: float = 30, overbought: float = 70) -> float:
        """
        计算RSI指标
        :param data: 价格数据
        :param period: 周期
        :param oversold: 超卖阈值
        :param overbought: 超买阈值
        :return: RSI指标字典
        """
        # 计算价格变动
        delta = data.diff()

        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 计算平均收益和平均损失
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # 计算RS
        rs = avg_gain / avg_loss

        # 计算RSI
        rsi = 100 - (100 / (1 + rs))

        # 获取最新值
        current_rsi = rsi.iloc[-1]
        # 为兼容现有单元测试，返回纯数值 RSI
        return float(current_rsi)

    @staticmethod
    def rsi_with_meta(data: pd.Series, period: int = 14,
                      oversold: float = 30, overbought: float = 70) -> Dict[str, float]:
        """
        带信号信息的 RSI 计算，供内部高级逻辑使用。
        对外简单使用请直接调用 rsi() 获取数值。
        """
        current_rsi = TechnicalIndicators.rsi(data, period=period, oversold=oversold, overbought=overbought)

        # 判断信号
        if current_rsi >= overbought:
            signal = SignalType.STRONG_SELL
            confidence = min(1.0, (current_rsi - overbought) / (100 - overbought))
        elif current_rsi <= oversold:
            signal = SignalType.STRONG_BUY
            confidence = min(1.0, (oversold - current_rsi) / oversold)
        elif current_rsi > 50:
            signal = SignalType.BUY
            confidence = (current_rsi - 50) / 50 * 0.5
        else:
            signal = SignalType.SELL
            confidence = (50 - current_rsi) / 50 * 0.5

        return {
            'rsi': current_rsi,
            'signal': signal,
            'confidence': confidence,
            'oversold': oversold,
            'overbought': overbought
        }

    @staticmethod
    def kdj(data: pd.Series,
            k_period: int = 9,
            d_period: int = 3,
            j_period: int = 3,
            oversold: float = 20,
            overbought: float = 80) -> Dict[str, float]:
        """
        计算KDJ指标
        :param data: 价格数据
        :param k_period: K值周期
        :param d_period: D值周期
        :param j_period: J值周期
        :param oversold: 超卖阈值
        :param overbought: 超买阈值
        :return: KDJ指标字典
        """
        # 计算RSV (未成熟随机值)
        low_min = data.rolling(window=k_period).min()
        high_max = data.rolling(window=k_period).max()

        rsv = (data - low_min) / (high_max - low_min) * 100

        # 计算K值
        k_values = []
        k_values.append(rsv.iloc[0] if not pd.isna(rsv.iloc[0]) else 50)
        for i in range(1, len(rsv)):
            if pd.isna(rsv.iloc[i]):
                k_values.append(k_values[-1])
            else:
                k_values.append((k_values[-1] * 2 / 3 + rsv.iloc[i] * 1 / 3))

        k = pd.Series(k_values, index=data.index)

        # 计算D值
        d = k.rolling(window=d_period).mean()

        # 计算J值
        j = 3 * k - 2 * d

        # 获取最新值
        current_k = k.iloc[-1]
        current_d = d.iloc[-1]
        current_j = j.iloc[-1]

        # 判断信号
        if current_j >= overbought:
            signal = SignalType.STRONG_SELL
            confidence = min(1.0, (current_j - overbought) / (100 - overbought))
        elif current_j <= oversold:
            signal = SignalType.STRONG_BUY
            confidence = min(1.0, (oversold - current_j) / oversold)
        elif current_k > current_d and current_j > 50:
            signal = SignalType.BUY
            confidence = min(0.8, (current_j - 50) / 50)
        elif current_k < current_d and current_j < 50:
            signal = SignalType.SELL
            confidence = min(0.8, (50 - current_j) / 50)
        else:
            signal = SignalType.HOLD
            confidence = 0.3

        return {
            'k': current_k,
            'd': current_d,
            'j': current_j,
            'signal': signal,
            'confidence': confidence,
            'oversold': oversold,
            'overbought': overbought
        }

    @staticmethod
    def bollinger_bands(data: pd.Series,
                        period: int = 20,
                        std_dev: float = 2) -> Dict[str, float]:
        """
        计算布林带
        :param data: 价格数据
        :param period: 移动平均周期
        :param std_dev: 标准差倍数
        :return: 布林带指标字典
        """
        # 计算中轨(简单移动平均线)
        middle_band = data.rolling(window=period).mean()

        # 计算标准差
        std = data.rolling(window=period).std()

        # 计算上轨和下轨
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)

        # 获取最新值
        current_price = data.iloc[-1]
        current_middle = middle_band.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_std = std.iloc[-1]

        # 计算带宽和位置
        bandwidth = (current_upper - current_lower) / current_middle if current_middle != 0 else 0
        position = (current_price - current_lower) / (current_upper - current_lower) if (current_upper - current_lower) != 0 else 0.5

        # 判断信号
        if current_price >= current_upper:
            signal = SignalType.STRONG_SELL
            confidence = min(1.0, (current_price - current_upper) / max(current_std, 1e-6))
        elif current_price <= current_lower:
            signal = SignalType.STRONG_BUY
            confidence = min(1.0, (current_lower - current_price) / max(current_std, 1e-6))
        elif position > 0.75:
            signal = SignalType.SELL
            confidence = (position - 0.75) / 0.25 * 0.7
        elif position < 0.25:
            signal = SignalType.BUY
            confidence = (0.25 - position) / 0.25 * 0.7
        else:
            signal = SignalType.HOLD
            confidence = 0.5

        return {
            'price': current_price,
            # 保留原有字段并增加兼容字段名，满足单测对 'upper'/'middle'/'lower' 的检查
            'upper_band': current_upper,
            'middle_band': current_middle,
            'lower_band': current_lower,
            'upper': current_upper,
            'middle': current_middle,
            'lower': current_lower,
            'bandwidth': bandwidth,
            'position': position,
            'signal': signal,
            'confidence': confidence,
            'period': period,
            'std_dev': std_dev
        }

    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame,
                                close_col: str = 'close',
                                config: Dict = None) -> List[IndicatorSignal]:
        """
        计算所有技术指标
        :param data: K线数据
        :param close_col: 收盘价格列名
        :param config: 配置字典
        :return: 指标信号列表
        """
        if config is None:
            config = {
                'ma': {'periods': [5, 10, 20, 60]},
                'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                'rsi': {'period': 14, 'oversold': 30, 'overbought': 70},
                'kdj': {'k_period': 9, 'd_period': 3, 'j_period': 3, 'oversold': 20, 'overbought': 80},
                'bollinger': {'period': 20, 'std_dev': 2}
            }

        price_series = data[close_col]
        indicators_signals = []

        # 计算MA信号
        ma_values = TechnicalIndicators.ma(price_series, **config['ma'])
        for period, ma_value in ma_values.items():
            if not pd.isna(ma_value):
                # 简单的MA信号判断
                current_price = price_series.iloc[-1]
                if current_price > ma_value:
                    signal = SignalType.BUY
                    confidence = min(0.6, (current_price - ma_value) / ma_value)
                else:
                    signal = SignalType.SELL
                    confidence = min(0.6, (ma_value - current_price) / ma_value)

                indicators_signals.append(IndicatorSignal(
                    name=f"MA{period}",
                    value=ma_value,
                    signal_type=signal,
                    confidence=confidence,
                    timestamp=datetime.now()
                ))

        # 计算MACD信号
        macd_result = TechnicalIndicators.macd(price_series, **config['macd'])
        if not any(pd.isna(v) for v in macd_result.values()):
            dif, dea, macd_hist = macd_result['dif'], macd_result['dea'], macd_result['macd_hist']

            # DIF与DEA的金叉死叉判断
            if dif > dea and macd_hist > 0:
                signal = SignalType.BUY
                confidence = min(0.8, abs(dif - dea) / abs(dif) if dif != 0 else 0.5)
            else:
                signal = SignalType.SELL
                confidence = min(0.8, abs(dif - dea) / abs(dif) if dif != 0 else 0.5)

            indicators_signals.append(IndicatorSignal(
                name="MACD",
                value=macd_hist,
                signal_type=signal,
                confidence=confidence,
                timestamp=datetime.now()
            ))

        # 计算RSI信号（使用带元信息的版本，保持对外 rsi() 简单数值接口）
        rsi_result = TechnicalIndicators.rsi_with_meta(price_series, **config['rsi'])
        indicators_signals.append(IndicatorSignal(
            name="RSI",
            value=rsi_result['rsi'],
            signal_type=rsi_result['signal'],
            confidence=rsi_result['confidence'],
            timestamp=datetime.now()
        ))

        # 计算KDJ信号
        kdj_result = TechnicalIndicators.kdj(price_series, **config['kdj'])
        indicators_signals.append(IndicatorSignal(
            name="KDJ",
            value=kdj_result['j'],
            signal_type=kdj_result['signal'],
            confidence=kdj_result['confidence'],
            timestamp=datetime.now()
        ))

        # 计算布林带信号
        bollinger_result = TechnicalIndicators.bollinger_bands(price_series, **config['bollinger'])
        indicators_signals.append(IndicatorSignal(
            name="Bollinger",
            value=bollinger_result['position'],
            signal_type=bollinger_result['signal'],
            confidence=bollinger_result['confidence'],
            timestamp=datetime.now()
        ))

        # 计算威廉指标信号
        wr_result = TechnicalIndicators.williams_r(price_series, period=14, oversold=-80, overbought=-20)
        indicators_signals.append(IndicatorSignal(
            name="WR",
            value=wr_result['wr'],
            signal_type=wr_result['signal'],
            confidence=wr_result['confidence'],
            timestamp=datetime.now()
        ))

        # 计算CCI信号
        cci_result = TechnicalIndicators.cci(price_series, period=20, oversold=-100, overbought=100)
        indicators_signals.append(IndicatorSignal(
            name="CCI",
            value=cci_result['cci'],
            signal_type=cci_result['signal'],
            confidence=cci_result['confidence'],
            timestamp=datetime.now()
        ))

        # 计算DMI信号
        dmi_result = TechnicalIndicators.dmi(price_series, period=14, adx_period=14)
        indicators_signals.append(IndicatorSignal(
            name="DMI",
            value=dmi_result['adx'],
            signal_type=dmi_result['signal'],
            confidence=dmi_result['confidence'],
            timestamp=datetime.now()
        ))

        # 添加ATR指标(不作为信号,但用于风险评估)
        atr_value = TechnicalIndicators.atr(price_series)
        indicators_signals.append(IndicatorSignal(
            name="ATR",
            value=atr_value,
            signal_type=SignalType.HOLD,  # ATR不是交易信号,仅用于风险评估
            confidence=0.0,
            timestamp=datetime.now()
        ))

        # 添加成交量分布信息
        volume_profile = TechnicalIndicators.volume_profile(data)
        if volume_profile['is_high_volume_area']:
            indicators_signals.append(IndicatorSignal(
                name="VolumeProfile",
                value=1.0,
                signal_type=SignalType.HOLD,  # 成交量密集区域是重要参考但不直接生成信号
                confidence=0.8,
                timestamp=datetime.now()
            ))

        return indicators_signals

    @staticmethod
    def williams_r(data: pd.Series, period: int = 14,
                   oversold: float = -80, overbought: float = -20) -> Dict[str, float]:
        """
        计算威廉指标
        :param data: 价格数据
        :param period: 周期
        :param oversold: 超卖阈值
        :param overbought: 超买阈值
        :return: 威廉指标字典
        """
        # 计算最高价和最低价
        high_max = data.rolling(window=period).max()
        low_min = data.rolling(window=period).min()

        # 计算威廉指标
        wr = (high_max - data) / (high_max - low_min) * -100

        # 获取最新值
        current_wr = wr.iloc[-1]

        # 判断信号
        if current_wr <= oversold:
            signal = SignalType.STRONG_BUY
            confidence = min(1.0, (oversold - current_wr) / abs(oversold))
        elif current_wr >= overbought:
            signal = SignalType.STRONG_SELL
            confidence = min(1.0, (current_wr - overbought) / abs(overbought))
        elif current_wr < -50:
            signal = SignalType.BUY
            confidence = min(0.7, (-50 - current_wr) / 50)
        else:
            signal = SignalType.SELL
            confidence = min(0.7, (current_wr + 50) / 50)

        return {
            'wr': current_wr,
            'signal': signal,
            'confidence': confidence,
            'oversold': oversold,
            'overbought': overbought
        }

    @staticmethod
    def cci(data: pd.Series, period: int = 20,
            oversold: float = -100, overbought: float = 100) -> Dict[str, float]:
        """
        计算顺势指标CCI
        :param data: 价格数据
        :param period: 周期
        :param oversold: 超卖阈值
        :param overbought: 超买阈值
        :return: CCI指标字典
        """
        # 计算典型价格
        typical_price = (data + data.shift(1) + data.shift(2)) / 3

        # 计算移动平均
        sma = typical_price.rolling(window=period).mean()

        # 计算平均绝对偏差
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))

        # 计算CCI
        cci = (typical_price - sma) / (0.015 * mad)

        # 获取最新值
        current_cci = cci.iloc[-1]

        # 判断信号
        if current_cci <= oversold:
            signal = SignalType.STRONG_BUY
            confidence = min(1.0, (oversold - current_cci) / abs(oversold))
        elif current_cci >= overbought:
            signal = SignalType.STRONG_SELL
            confidence = min(1.0, (current_cci - overbought) / overbought)
        elif current_cci > 0:
            signal = SignalType.BUY
            confidence = min(0.7, current_cci / overbought)
        else:
            signal = SignalType.SELL
            confidence = min(0.7, -current_cci / abs(oversold))

        return {
            'cci': current_cci,
            'signal': signal,
            'confidence': confidence,
            'oversold': oversold,
            'overbought': overbought
        }

    @staticmethod
    def dmi(data: pd.Series, period: int = 14,
            adx_period: int = 14) -> Dict[str, float]:
        """
        计算动向指标DMI
        :param data: 价格数据
        :param period: DI计算周期
        :param adx_period: ADX计算周期
        :return: DMI指标字典
        """
        # 计算价格变动
        high_diff = data.diff().abs()
        low_diff = data.diff().abs().shift(1)

        # 计算真实波幅TR
        tr = np.maximum(high_diff, low_diff)
        tr = tr.fillna(method='bfill').fillna(0)

        # 计算+DI和-DI
        plus_dm = data.diff().clip(lower=0)
        minus_dm = -data.diff().clip(upper=0)

        smooth_plus_dm = plus_dm.rolling(window=period).sum()
        smooth_minus_dm = minus_dm.rolling(window=period).sum()

        smooth_tr = tr.rolling(window=period).sum()

        pdi = 100 * smooth_plus_dm / smooth_tr
        ndi = 100 * smooth_minus_dm / smooth_tr

        # 计算ADX
        dx = 100 * abs(pdi - ndi) / (pdi + ndi)
        adx = dx.rolling(window=adx_period).mean()

        # 获取最新值
        current_pdi = pdi.iloc[-1]
        current_ndi = ndi.iloc[-1]
        current_adx = adx.iloc[-1]

        # 判断信号
        if current_adx > 25:
            if current_pdi > current_ndi:
                signal = SignalType.STRONG_BUY
                confidence = min(1.0, current_adx / 50)
            else:
                signal = SignalType.STRONG_SELL
                confidence = min(1.0, current_adx / 50)
        elif current_pdi > current_ndi:
            signal = SignalType.BUY
            confidence = min(0.7, current_adx / 50)
        else:
            signal = SignalType.SELL
            confidence = min(0.7, current_adx / 50)

        return {
            'pdi': current_pdi,
            'ndi': current_ndi,
            'adx': current_adx,
            'signal': signal,
            'confidence': confidence,
            'period': period
        }

    @staticmethod
    def atr(data: pd.Series, period: int = 14) -> float:
        """
        计算平均真实波幅ATR
        :param data: 价格数据
        :param period: 周期
        :return: ATR值
        """
        high_diff = data.diff().abs()
        low_diff = data.diff().abs().shift(1)

        # 计算真实波幅TR
        tr = np.maximum(high_diff, low_diff)
        tr = tr.fillna(method='bfill').fillna(0)

        # 计算ATR
        atr = tr.rolling(window=period).mean().iloc[-1]

        return atr

    @staticmethod
    def volume_profile(data: pd.DataFrame, price_col: str = 'close',
                       volume_col: str = 'volume') -> Dict[str, float]:
        """
        计算成交量分布
        :param data: K线数据
        :param price_col: 价格列名
        :param volume_col: 成交量列名
        :return: 成交量分布字典
        """
        # 计算价格区间
        price_min = data[price_col].min()
        price_max = data[price_col].max()

        # 将价格分成10个区间
        price_bins = np.linspace(price_min, price_max, 11)

        # 计算每个区间的成交量
        volume_distribution = {}
        for i in range(len(price_bins) - 1):
            mask = (data[price_col] >= price_bins[i]) & (data[price_col] < price_bins[i + 1])
            volume = data.loc[mask, volume_col].sum()
            volume_distribution[f"{price_bins[i]:.2f}-{price_bins[i + 1]:.2f}"] = volume

        # 找出最大成交量区间
        max_volume_interval = max(volume_distribution, key=volume_distribution.get)
        max_volume = volume_distribution[max_volume_interval]

        # 计算当前价格在哪个区间
        current_price = data[price_col].iloc[-1]
        current_interval = None
        for i in range(len(price_bins) - 1):
            if current_price >= price_bins[i] and current_price < price_bins[i + 1]:
                current_interval = f"{price_bins[i]:.2f}-{price_bins[i + 1]:.2f}"
                break

        # 判断价格是否在密集交易区域
        is_high_volume_area = current_interval == max_volume_interval

        return {
            'current_price_interval': current_interval,
            'max_volume_interval': max_volume_interval,
            'is_high_volume_area': is_high_volume_area,
            'volume_distribution': volume_distribution
        }