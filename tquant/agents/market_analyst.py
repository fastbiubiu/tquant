"""
市场分析师Agent
负责技术分析和生成交易信号
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np

from tquant.config import get_config, Config
from tquant.utils.indicators import TechnicalIndicators
from tquant.utils.signals import TradingSignal, SignalType, IndicatorSignal, SignalGenerator
from tquant.utils.tqsdk_interface import TqSdkInterface

logger = logging.getLogger(__name__)


class MarketAnalyst:
    """市场分析师Agent"""

    def __init__(self, config_path: str = None):
        """初始化分析师"""
        self.config: Config = get_config()
        self.tqsdk = TqSdkInterface(config_path)
        self.indicators_config = self.config.indicators.model_dump() if self.config.indicators else {}
        self.signals_config = {}

    def connect(self, backtest: Optional[bool] = None, demo: Optional[bool] = None) -> bool:
        """连接API（参数不填则完全使用配置文件中的 tqsdk 设置）"""
        return self.tqsdk.connect(backtest=backtest, demo=demo)

    def analyze_symbol(self, symbol: str, kline_count: int = 100) -> Optional[TradingSignal]:
        """
        分析单个品种
        :param symbol: 交易品种
        :param kline_count: K线数据数量
        :return: 交易信号
        """
        try:
            # 获取K线数据
            kline_data = self.tqsdk.get_kline_data(symbol, count=kline_count)

            if kline_data.empty:
                logger.error(f"获取{symbol}的K线数据失败")
                return None

            # 获取当前价格
            current_price = kline_data['close'].iloc[-1]
            logger.info(f"{symbol} 当前价格: {current_price:.2f}")

            # 计算技术指标
            indicators = TechnicalIndicators.calculate_all_indicators(
                kline_data,
                config=self.indicators_config
            )

            if not indicators:
                logger.warning(f"{symbol} 未生成有效的技术指标")
                return None

            # 生成信号说明
            reasoning = self._generate_reasoning(symbol, indicators, current_price)

            # 创建交易信号
            signal = SignalGenerator.create_trading_signal(
                symbol=symbol,
                price=current_price,
                indicators=indicators,
                reasoning=reasoning,
                thresholds=self.signals_config.get('thresholds', None)
            )

            logger.info(f"{symbol} 分析完成: {signal.signal_type.value} (信心度: {signal.confidence:.2f})")
            return signal

        except Exception as e:
            logger.error(f"分析{symbol}时发生错误: {e}")
            return None

    def analyze_multiple_symbols(self, symbols: List[str], kline_count: int = 100) -> List[TradingSignal]:
        """
        分析多个品种
        :param symbols: 交易品种列表
        :param kline_count: K线数据数量
        :return: 交易信号列表
        """
        signals = []
        analysis_summary = []

        for symbol in symbols:
            logger.info(f"开始分析 {symbol}...")
            signal = self.analyze_symbol(symbol, kline_count)

            if signal:
                signals.append(signal)
                analysis_summary.append(f"✅ {symbol}: {signal.signal_type.value} (信心度: {signal.confidence:.2f})")
                logger.info(f"✅ {symbol} 分析完成")
            else:
                analysis_summary.append(f"❌ {symbol}: 分析失败")
                logger.warning(f"❌ {symbol} 分析失败")

        # 按信心度排序
        signals.sort(key=lambda x: x.confidence, reverse=True)

        # 输出分析摘要
        logger.info("\n=== 分析摘要 ===")
        for summary in analysis_summary:
            logger.info(summary)

        return signals

    def analyze_market_trend(self, symbol: str, kline_count: int = 100) -> Dict[str, Any]:
        """
        分析市场趋势
        :param symbol: 交易品种
        :param kline_count: K线数据数量
        :return: 趋势分析结果
        """
        try:
            # 获取K线数据
            kline_data = self.tqsdk.get_kline_data(symbol, count=kline_count)

            if kline_data.empty:
                return {'error': '无法获取K线数据'}

            # 计算短期和长期移动平均线
            short_ma = kline_data['close'].rolling(window=5).mean().iloc[-1]
            medium_ma = kline_data['close'].rolling(window=20).mean().iloc[-1]
            long_ma = kline_data['close'].rolling(window=60).mean().iloc[-1]

            current_price = kline_data['close'].iloc[-1]

            # 判断趋势
            trend = {
                'current_price': current_price,
                'short_ma': short_ma,
                'medium_ma': medium_ma,
                'long_ma': long_ma
            }

            # 趋势判断逻辑
            if current_price > short_ma > medium_ma > long_ma:
                trend['direction'] = 'STRONG_UPWARD'
                trend['strength'] = '强'
            elif current_price > medium_ma > long_ma:
                trend['direction'] = 'UPWARD'
                trend['strength'] = '中'
            elif short_ma > current_price > medium_ma:
                trend['direction'] = 'WEAK_UPWARD'
                trend['strength'] = '弱'
            elif long_ma > medium_ma > current_price:
                trend['direction'] = 'DOWNWARD'
                trend['strength'] = '中'
            elif current_price < short_ma < medium_ma < long_ma:
                trend['direction'] = 'STRONG_DOWNWARD'
                trend['strength'] = '强'
            else:
                trend['direction'] = 'SIDEWAYS'
                trend['strength'] = '横盘'

            return trend

        except Exception as e:
            logger.error(f"分析{symbol}市场趋势失败: {e}")
            return {'error': str(e)}

    def calculate_volatility(self, symbol: str, kline_count: int = 100) -> Dict[str, float]:
        """
        计算波动率
        :param symbol: 交易品种
        :param kline_count: K线数据数量
        :return: 波动率信息
        """
        try:
            # 获取K线数据
            kline_data = self.tqsdk.get_kline_data(symbol, count=kline_count)

            if kline_data.empty:
                return {'error': '无法获取K线数据'}

            # 计算对数收益率
            returns = np.log(kline_data['close'] / kline_data['close'].shift(1)).dropna()

            # 计算波动率
            volatility = returns.std() * np.sqrt(252)  # 年化波动率
            volatility_20 = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)  # 20日波动率

            # 计算ATR(平均真实波幅)
            high_low = kline_data['high'] - kline_data['low']
            high_close = np.abs(kline_data['high'] - kline_data['close'].shift(1))
            low_close = np.abs(kline_data['low'] - kline_data['close'].shift(1))

            tr = np.maximum(high_low, high_close, low_close)
            atr = tr.rolling(window=14).mean().iloc[-1]

            # 判断波动水平
            volatility_level = 'LOW'
            if volatility > 0.3:
                volatility_level = 'HIGH'
            elif volatility > 0.15:
                volatility_level = 'MEDIUM'

            return {
                'volatility': volatility,
                'volatility_20': volatility_20,
                'atr': atr,
                'volatility_level': volatility_level,
                'confidence': min(volatility / 0.5, 1.0)  # 波动率越高,信心度越高
            }

        except Exception as e:
            logger.error(f"计算{symbol}波动率失败: {e}")
            return {'error': str(e)}

    def get_signal_strength_analysis(self, signals: List[TradingSignal]) -> Dict[str, Any]:
        """
        分析信号强度
        :param signals: 交易信号列表
        :return: 信号强度分析
        """
        if not signals:
            return {'error': '无交易信号'}

        # 统计信号类型
        signal_counts = {}
        total_confidence = 0
        buy_signals = []
        sell_signals = []

        for signal in signals:
            signal_type = signal.signal_type.value
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            total_confidence += signal.confidence

            if signal.signal_type.value in ['强烈买入', '买入']:
                buy_signals.append(signal)
            elif signal.signal_type.value in ['强烈卖出', '卖出']:
                sell_signals.append(signal)

        # 计算整体市场情绪
        market_sentiment = 'NEUTRAL'
        if len(buy_signals) > len(sell_signals) * 1.5:
            market_sentiment = 'BULLISH'
        elif len(sell_signals) > len(buy_signals) * 1.5:
            market_sentiment = 'BEARISH'

        # 计算平均信心度
        avg_confidence = total_confidence / len(signals)

        return {
            'total_signals': len(signals),
            'signal_distribution': signal_counts,
            'average_confidence': avg_confidence,
            'market_sentiment': market_sentiment,
            'buy_signals_count': len(buy_signals),
            'sell_signals_count': len(sell_signals),
            'strongest_signal': signals[0] if signals else None
        }

    def _generate_reasoning(self, symbol: str, indicators: List[IndicatorSignal], price: float) -> str:
        """生成分析推理过程"""
        reasoning = f"\n=== {symbol} 技术分析推理 ===\n"
        reasoning += f"当前价格: {price:.2f}\n\n"

        # 统计各指标信号
        buy_signals = [ind for ind in indicators if ind.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]]
        sell_signals = [ind for ind in indicators if ind.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]]
        hold_signals = [ind for ind in indicators if ind.signal_type == SignalType.HOLD]

        reasoning += f"买入信号: {len(buy_signals)} 个\n"
        reasoning += f"卖出信号: {len(sell_signals)} 个\n"
        reasoning += f"持有信号: {len(hold_signals)} 个\n\n"

        # 详细分析各指标
        reasoning += "指标详情:\n"
        for indicator in indicators:
            reasoning += f"• {indicator.name}: {indicator.signal_type.value} "
            reasoning += f"(值: {indicator.value:.2f}, 信心度: {indicator.confidence:.2f})\n"

        # 综合分析
        reasoning += f"\n综合分析:\n"

        if len(buy_signals) > len(sell_signals):
            reasoning += "• 多数指标显示买入机会,建议关注入场时机\n"
            if len(buy_signals) >= 3:
                reasoning += "• 多个指标达成共识,信号较为可靠\n"
        elif len(sell_signals) > len(buy_signals):
            reasoning += "• 多数指标显示卖出压力,建议谨慎操作\n"
            if len(sell_signals) >= 3:
                reasoning += "• 多个指标达成共识,信号较为可靠\n"
        else:
            reasoning += "• 指标信号分歧较大,建议继续观察或轻仓试探\n"

        # 风险提示
        reasoning += f"\n风险提示:\n"
        reasoning += "• 技术指标仅供参考,需结合基本面分析\n"
        reasoning += "• 建议设置止损止盈,控制风险\n"
        reasoning += "• 市场波动较大时,信号可能失效\n"

        return reasoning

    def get_market_summary(self, signals: List[TradingSignal]) -> str:
        """生成市场分析摘要"""
        if not signals:
            return "暂无有效的市场信号"

        summary = f"\n=== 市场分析摘要 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n"

        # 统计信号类型
        signal_types = {}
        total_confidence = 0
        buy_signals = []
        sell_signals = []

        for signal in signals:
            signal_type = signal.signal_type.value
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
            total_confidence += signal.confidence

            if signal.signal_type.value in ['强烈买入', '买入']:
                buy_signals.append(signal)
            elif signal.signal_type.value in ['强烈卖出', '卖出']:
                sell_signals.append(signal)

        summary += "信号统计:\n"
        for signal_type, count in signal_types.items():
            summary += f"• {signal_type}: {count} 个\n"

        summary += f"\n市场情绪:\n"
        market_sentiment = '中性'
        if len(buy_signals) > len(sell_signals) * 1.5:
            market_sentiment = '看涨'
            summary += "• 整体市场情绪偏向看涨\n"
        elif len(sell_signals) > len(buy_signals) * 1.5:
            market_sentiment = '看跌'
            summary += "• 整体市场情绪偏向看跌\n"
        else:
            summary += "• 整体市场情绪相对中性\n"

        # 计算平均信心度
        avg_confidence = total_confidence / len(signals)
        summary += f"• 平均信号信心度: {avg_confidence:.2f}\n"

        # 显示最强信号
        if signals:
            summary += f"\n最强信号:\n"
            top_signal = signals[0]
            summary += f"• 品种: {top_signal.symbol}\n"
            summary += f"• 信号: {top_signal.signal_type.value}\n"
            summary += f"• 信心度: {top_signal.confidence:.2f}\n"
            summary += f"• 价格: {top_signal.price:.2f}\n"
            summary += f"• 操作建议: {top_signal.get_action() or '等待'}\n"

        # 显示趋势分析
        if signals:
            summary += f"\n趋势分析:\n"
            # 分析第一个品种的趋势作为参考
            symbol = signals[0].symbol
            trend_analysis = self.analyze_market_trend(symbol)
            if 'error' not in trend_analysis:
                summary += f"• {symbol} 趋势方向: {trend_analysis.get('direction', '未知')}\n"
                summary += f"• 趋势强度: {trend_analysis.get('strength', '未知')}\n"

        # 显示波动率分析
        if signals:
            volatility_analysis = self.calculate_volatility(signals[0].symbol)
            if 'error' not in volatility_analysis:
                summary += f"\n波动率分析:\n"
                summary += f"• 波动水平: {volatility_analysis.get('volatility_level', '未知')}\n"
                summary += f"• ATR值: {volatility_analysis.get('atr', 0):.2f}\n"

        # 风险提示
        summary += f"\n风险提示:\n"
        summary += "• 技术指标仅供参考,需结合基本面分析\n"
        summary += "• 建议设置止损止盈,控制风险\n"
        summary += "• 市场波动较大时,信号可能失效\n"
        summary += f"• 当前波动水平: {'高' if 'HIGH' in [volatility_analysis.get('volatility_level', '')] else '中等' if 'MEDIUM' in [volatility_analysis.get('volatility_level', '')] else '低'}\n"

        return summary

    def get_detailed_analysis(self, symbol: str) -> str:
        """获取详细分析报告"""
        try:
            # 获取K线数据
            kline_data = self.tqsdk.get_kline_data(symbol, count=100)

            if kline_data.empty:
                return f"无法获取{symbol}的数据"

            # 计算指标
            indicators = TechnicalIndicators.calculate_all_indicators(kline_data)

            # 分析趋势
            trend = self.analyze_market_trend(symbol)

            # 计算波动率
            volatility = self.calculate_volatility(symbol)

            # 生成报告
            report = f"\n=== {symbol} 详细分析报告 ===\n\n"

            # 基本信息
            report += f"当前价格: {kline_data['close'].iloc[-1]:.2f}\n"
            report += f"24小时最高: {kline_data['high'].iloc[-1]:.2f}\n"
            report += f"24小时最低: {kline_data['low'].iloc[-1]:.2f}\n\n"

            # 趋势分析
            if 'error' not in trend:
                report += f"趋势分析:\n"
                report += f"• 方向: {trend.get('direction', '未知')}\n"
                report += f"• 强度: {trend.get('strength', '未知')}\n"
                report += f"• 短期均线: {trend.get('short_ma', 0):.2f}\n"
                report += f"• 中期均线: {trend.get('medium_ma', 0):.2f}\n"
                report += f"• 长期均线: {trend.get('long_ma', 0):.2f}\n\n"

            # 波动率分析
            if 'error' not in volatility:
                report += f"波动率分析:\n"
                report += f"• 年化波动率: {volatility.get('volatility', 0):.2%}\n"
                report += f"• 20日波动率: {volatility.get('volatility_20', 0):.2%}\n"
                report += f"• ATR(14): {volatility.get('atr', 0):.2f}\n"
                report += f"• 波动水平: {volatility.get('volatility_level', '未知')}\n\n"

            # 技术指标信号
            report += "技术指标信号:\n"
            for indicator in indicators:
                if indicator.signal_type != '持有':  # 只显示非持有信号
                    report += f"• {indicator.name}: {indicator.signal_type.value} "
                    report += f"(值: {indicator.value:.2f}, 信心度: {indicator.confidence:.2f})\n"

            # 风险提示
            report += f"\n风险提示:\n"
            if 'HIGH' in volatility.get('volatility_level', ''):
                report += "• 警告：高波动环境,交易风险较大\n"
            elif 'MEDIUM' in volatility.get('volatility_level', ''):
                report += "• 注意：中等波动环境,需谨慎交易\n"

            report += "• 建议设置止损,控制单笔亏损\n"
            report += "• 关注成交量变化,确认信号可靠性\n"

            return report

        except Exception as e:
            logger.error(f"生成{symbol}详细分析失败: {e}")
            return f"生成详细分析失败: {e}"

    def close(self):
        """关闭连接"""
        self.tqsdk.close()