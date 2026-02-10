"""
量化交易工作流
使用LangGraph编排Market Analyst和Trader Agent
"""

import logging
from datetime import datetime
from typing import Dict, List, TypedDict, Optional

from langgraph.graph import StateGraph, END

from tquant.agents.market_analyst import MarketAnalyst
from tquant.agents.trader import Trader
from tquant.config import get_config, Config
from tquant.utils.signals import TradingSignal, SignalType

logger = logging.getLogger(__name__)


class TradingState(TypedDict):
    """交易状态"""
    symbols: List[str]
    current_symbol: Optional[str]
    market_signals: List[TradingSignal]
    current_signal: Optional[TradingSignal]
    portfolio_summary: str
    risk_metrics: Dict
    execution_results: List[Dict]
    timestamp: str
    messages: List[Dict]


class TradingWorkflow:
    """量化交易工作流"""

    def __init__(self, config_path: str = None):
        """初始化工作流"""
        self.config: Config = get_config()

        # 初始化Agent
        self.market_analyst = MarketAnalyst(config_path)
        self.trader = Trader(config_path)

        # 创建工作流
        self.workflow = self._create_workflow()

        # 初始化状态
        self.state = self._init_state()

    def _init_state(self) -> TradingState:
        """初始化状态"""
        return {
            'symbols': self.config.trading.symbols,
            'current_symbol': None,
            'market_signals': [],
            'current_signal': None,
            'portfolio_summary': '',
            'risk_metrics': {},
            'execution_results': [],
            'timestamp': datetime.now().isoformat(),
            'messages': []
        }

    def _create_workflow(self) -> StateGraph:
        """创建工作流"""
        # 创建状态图
        workflow = StateGraph(TradingState)

        # 添加节点
        workflow.add_node("analyze_market", self.analyze_market)
        workflow.add_node("evaluate_signals", self.evaluate_signals)
        workflow.add_node("execute_trades", self.execute_trades)
        workflow.add_node("update_portfolio", self.update_portfolio)
        workflow.add_node("monitor_risk", self.monitor_risk)

        # 设置流程
        workflow.set_entry_point("analyze_market")

        # 添加边
        workflow.add_edge("analyze_market", "evaluate_signals")
        workflow.add_edge("evaluate_signals", "execute_trades")
        workflow.add_edge("execute_trades", "update_portfolio")
        workflow.add_edge("update_portfolio", "monitor_risk")
        workflow.add_edge("monitor_risk", END)

        return workflow.compile()

    def analyze_market(self, state: TradingState) -> TradingState:
        """分析市场"""
        logger.info("=== 开始分析市场 ===")

        symbols = state['symbols']
        market_signals = []

        # 分析所有品种
        for symbol in symbols:
            logger.info(f"分析品种: {symbol}")

            try:
                signal = self.market_analyst.analyze_symbol(symbol)
                if signal:
                    market_signals.append(signal)
                    state['messages'].append({
                        'type': 'market_analysis',
                        'symbol': symbol,
                        'signal': signal.signal_type.value,
                        'confidence': signal.confidence,
                        'timestamp': datetime.now().isoformat()
                    })
                    logger.info(f"✅ {symbol}: {signal.signal_type.value} (信心度: {signal.confidence:.2f})")
                else:
                    logger.warning(f"❌ {symbol}: 分析失败")
            except Exception as e:
                logger.error(f"分析{symbol}时出错: {e}")
                state['messages'].append({
                    'type': 'error',
                    'symbol': symbol,
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                })

        # 按信心度排序
        market_signals.sort(key=lambda x: x.confidence, reverse=True)

        state['market_signals'] = market_signals
        state['timestamp'] = datetime.now().isoformat()

        logger.info(f"市场分析完成,共生成{len(market_signals)}个信号")
        return state

    def evaluate_signals(self, state: TradingState) -> TradingState:
        """评估信号"""
        logger.info("=== 评估交易信号 ===")

        market_signals = state['market_signals']
        selected_signals = []

        if not market_signals:
            state['messages'].append({
                'type': 'warning',
                'message': '没有有效的交易信号',
                'timestamp': datetime.now().isoformat()
            })
            return state

        # 信号筛选逻辑
        for signal in market_signals:
            # 过滤信心度低的信号
            if signal.confidence < 0.3:
                logger.info(f"跳过{signal.symbol},信心度过低: {signal.confidence:.2f}")
                continue

            # 检查信号是否强烈
            if signal.signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
                selected_signals.append(signal)
                logger.info(f"✅ 选择{signal.symbol}：{signal.signal_type.value}")
            elif signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                # 对于一般信号,需要有其他指标支持
                supporting_indicators = sum(1 for ind in signal.indicators
                                        if ind.signal_type == signal.signal_type)
                if supporting_indicators >= 2:
                    selected_signals.append(signal)
                    logger.info(f"✅ 选择{signal.symbol}：{signal.signal_type.value} ({supporting_indicators}个指标支持)")
            else:
                logger.info(f"跳过{signal.symbol}：{signal.signal_type.value}")

        state['market_signals'] = selected_signals
        state['timestamp'] = datetime.now().isoformat()

        logger.info(f"信号评估完成,选择{len(selected_signals)}个信号执行")
        return state

    def execute_trades(self, state: TradingState) -> TradingState:
        """执行交易"""
        logger.info("=== 执行交易 ===")

        market_signals = state['market_signals']
        execution_results = []

        if not market_signals:
            state['messages'].append({
                'type': 'info',
                'message': '没有需要执行的交易信号',
                'timestamp': datetime.now().isoformat()
            })
            return state

        # 执行交易
        trading_result = self.trader.start_trading(market_signals)
        execution_results.extend(trading_result['results'])

        # 记录交易结果
        for result in trading_result['results']:
            state['messages'].append({
                'type': 'trade_execution',
                'symbol': result.get('symbol', '未知'),
                'success': result['success'],
                'message': result.get('message', ''),
                'timestamp': datetime.now().isoformat()
            })

        # 记录风险管理结果
        if trading_result.get('risk_actions'):
            for action in trading_result['risk_actions']:
                state['messages'].append({
                    'type': 'risk_management',
                    'action': action.get('type', '未知'),
                    'symbol': action.get('symbol', '未知'),
                    'timestamp': datetime.now().isoformat()
                })

        # 记录盈亏
        if trading_result.get('total_profit') is not None:
            state['messages'].append({
                'type': 'profit_loss',
                'total_profit': trading_result['total_profit'],
                'timestamp': datetime.now().isoformat()
            })

        state['execution_results'] = execution_results
        state['timestamp'] = datetime.now().isoformat()

        logger.info(f"交易执行完成,共执行{len(execution_results)}笔交易")
        return state

    def update_portfolio(self, state: TradingState) -> TradingState:
        """更新投资组合信息"""
        logger.info("=== 更新投资组合 ===")

        try:
            # 获取投资组合摘要
            portfolio_summary = self.trader.get_portfolio_summary()
            state['portfolio_summary'] = portfolio_summary

            # 获取风险指标
            risk_metrics = self.trader.get_risk_metrics()
            state['risk_metrics'] = risk_metrics

            state['messages'].append({
                'type': 'portfolio_update',
                'message': '投资组合信息已更新',
                'timestamp': datetime.now().isoformat()
            })

            logger.info("投资组合信息更新完成")
        except Exception as e:
            logger.error(f"更新投资组合失败: {e}")
            state['messages'].append({
                'type': 'error',
                'message': f"更新投资组合失败: {str(e)}",
                'timestamp': datetime.now().isoformat()
            })

        return state

    def monitor_risk(self, state: TradingState) -> TradingState:
        """监控风险"""
        logger.info("=== 监控风险 ===")

        risk_metrics = state['risk_metrics']

        # 检查风险水平
        risk_ratio = risk_metrics.get('risk_ratio', 0)

        if risk_ratio > 0.8:
            state['messages'].append({
                'type': 'high_risk',
                'message': f'风险度过高: {risk_ratio:.2%}',
                'timestamp': datetime.now().isoformat()
            })
            logger.warning(f"⚠️ 风险警告: {risk_ratio:.2%}")
        elif risk_ratio > 0.6:
            state['messages'].append({
                'type': 'medium_risk',
                'message': f'风险度较高: {risk_ratio:.2%}',
                'timestamp': datetime.now().isoformat()
            })
            logger.warning(f"⚠️ 风险提示: {risk_ratio:.2%}")

        # 检查持仓数量
        position_count = risk_metrics.get('position_count', 0)
        max_positions = self.config.get('risk', {}).get('max_positions', 5)

        if position_count > max_positions:
            state['messages'].append({
                'type': 'position_warning',
                'message': f'持仓数量过多: {position_count} > {max_positions}',
                'timestamp': datetime.now().isoformat()
            })
            logger.warning(f"⚠️ 持仓警告: {position_count} > {max_positions}")

        state['timestamp'] = datetime.now().isoformat()
        logger.info("风险监控完成")
        return state

    def run(self, symbols: List[str] = None) -> Dict:
        """运行工作流"""
        # 更新状态
        if symbols:
            self.state['symbols'] = symbols

        logger.info("=" * 60)
        logger.info("开始运行量化交易工作流")
        logger.info("=" * 60)

        # 运行工作流
        result = self.workflow.invoke(self.state)

        # 生成总结报告
        summary = self._generate_summary(result)

        return {
            'state': result,
            'summary': summary,
            'messages': result['messages']
        }

    def _generate_summary(self, state: TradingState) -> str:
        """生成总结报告"""
        summary = f"\n=== 交易工作流总结报告 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n"

        # 基本统计
        summary += f"基本信息:\n"
        summary += f"• 交易品种数: {len(state['symbols'])}\n"
        summary += f"• 生成信号数: {len(state['market_signals'])}\n"
        summary += f"• 执行交易数: {len(state['execution_results'])}\n"
        summary += f"• 持仓数量: {state['risk_metrics'].get('position_count', 0)}\n\n"

        # 交易结果
        successful_trades = sum(1 for r in state['execution_results'] if r.get('success', False))
        if successful_trades > 0:
            summary += f"交易结果:\n"
            summary += f"• 成功交易: {successful_trades}\n"
            summary += f"• 成功率: {successful_trades/len(state['execution_results'])*100:.1f}%\n\n"

        # 风险状况
        risk_ratio = state['risk_metrics'].get('risk_ratio', 0)
        summary += f"风险状况:\n"
        summary += f"• 风险度: {risk_ratio:.2%}\n"
        if risk_ratio > 0.8:
            summary += "• 状态: 高风险 ⚠️\n"
        elif risk_ratio > 0.6:
            summary += "• 状态: 中等风险 ⚡\n"
        else:
            summary += "• 状态: 风险可控 ✅\n\n"

        # 盈亏状况
        total_profit = 0
        for msg in state['messages']:
            if msg.get('type') == 'profit_loss':
                total_profit = msg.get('total_profit', 0)
                break

        summary += f"盈亏状况:\n"
        summary += f"• 总盈亏: {total_profit:.2f}\n"
        if total_profit > 0:
            summary += "• 状态: 盈利 📈\n"
        elif total_profit < 0:
            summary += "• 状态: 亏损 📉\n"
        else:
            summary += "• 状态: 平衡 ➖\n\n"

        # 重要消息
        summary += f"重要消息:\n"
        for msg in state['messages'][-5:]:  # 最近5条消息
            if msg.get('type') in ['error', 'warning', 'high_risk', 'medium_risk']:
                summary += f"• {msg['message']}\n"

        return summary

    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            'symbols': self.state['symbols'],
            'market_signals_count': len(self.state['market_signals']),
            'execution_results_count': len(self.state['execution_results']),
            'position_count': len(self.state['risk_metrics'].get('position_risks', [])),
            'last_update': self.state['timestamp']
        }

    def reset(self):
        """重置工作流"""
        self.state = self._init_state()
        logger.info("工作流已重置")