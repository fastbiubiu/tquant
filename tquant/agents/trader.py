"""
交易执行器Agent
负责执行交易信号和风险管理
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

from tquant.config import get_config, Config
from tquant.utils.signals import TradingSignal, SignalType
from tquant.utils.tqsdk_interface import TqSdkInterface

logger = logging.getLogger(__name__)


class Trader:
    """交易执行器Agent"""

    def __init__(self, config_path: str = None):
        """初始化交易器"""
        self.config: Config = get_config()
        self.tqsdk = TqSdkInterface(config_path)
        self.positions = {}
        self.orders = {}
        self.trade_history = []
        self.is_running = False

        # 交易配置
        self.symbols = self.config.trading.symbols
        self.max_position_ratio = self.config.trading.account.max_position_ratio
        self.max_loss_ratio = self.config.trading.risk.max_loss_ratio
        self.stop_loss_ratio = self.config.trading.risk.stop_loss_ratio
        self.take_profit_ratio = self.config.trading.risk.take_profit_ratio

    def connect(self, backtest: Optional[bool] = None, demo: Optional[bool] = None) -> bool:
        """连接API（参数不填则完全使用配置文件中的 tqsdk 设置）"""
        return self.tqsdk.connect(backtest=backtest, demo=demo)

    def execute_signal(self, signal: TradingSignal, account_info: Dict = None) -> Dict:
        """
        执行交易信号
        :param signal: 交易信号
        :param account_info: 账户信息
        :return: 执行结果
        """
        try:
            result = {
                'symbol': signal.symbol,
                'action': None,
                'success': False,
                'message': '',
                'order_id': None
            }

            # 检查信号是否需要执行
            if not signal.action_required:
                result['message'] = '信号不需要执行操作'
                return result

            # 获取账户信息
            if account_info is None:
                account_info = self.tqsdk.get_account_info()

            # 根据信号类型执行相应操作
            if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
                result = self._execute_buy(signal, account_info)
            elif signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
                result = self._execute_sell(signal, account_info)

            # 记录交易
            if result['success']:
                self._record_trade(signal, result)

            return result

        except Exception as e:
            logger.error(f"执行{signal.symbol}交易信号失败: {e}")
            return {
                'symbol': signal.symbol,
                'action': None,
                'success': False,
                'message': str(e),
                'order_id': None
            }

    def _execute_buy(self, signal: TradingSignal, account_info: Dict) -> Dict:
        """执行买入操作"""
        try:
            # 计算交易数量
            available = account_info.get('available', 0)
            max_amount = available * self.max_position_ratio

            # 简单交易逻辑：固定金额交易
            trade_amount = max_amount * 0.1  # 使用10%的可用资金
            price = signal.price
            volume = int(trade_amount / price / 100) * 100  # 按手数计算

            if volume <= 0:
                return {
                    'symbol': signal.symbol,
                    'action': 'BUY',
                    'success': False,
                    'message': '资金不足,无法买入',
                    'order_id': None
                }

            # 执行买入
            order = self.tqsdk.place_order(signal.symbol, 'BUY', 'OPEN', volume)

            if order.get('order_id'):
                self.positions[signal.symbol] = {
                    'direction': 'LONG',
                    'volume': volume,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'stop_loss': price * (1 - self.stop_loss_ratio),
                    'take_profit': price * (1 + self.take_profit_ratio)
                }

                return {
                    'symbol': signal.symbol,
                    'action': 'BUY',
                    'success': True,
                    'message': '买入订单已提交',
                    'order_id': order['order_id'],
                    'volume': volume,
                    'price': price
                }
            else:
                return {
                    'symbol': signal.symbol,
                    'action': 'BUY',
                    'success': False,
                    'message': '买入订单失败',
                    'order_id': None
                }

        except Exception as e:
            logger.error(f"执行买入操作失败: {e}")
            return {
                'symbol': signal.symbol,
                'action': 'BUY',
                'success': False,
                'message': str(e),
                'order_id': None
            }

    def _execute_sell(self, signal: TradingSignal, account_info: Dict) -> Dict:
        """执行卖出操作"""
        try:
            # 检查是否有持仓
            if signal.symbol not in self.positions:
                # 如果没有持仓,执行做空
                return self._execute_short(signal, account_info)
            else:
                # 有持仓,平仓
                return self._close_position(signal.symbol)

        except Exception as e:
            logger.error(f"执行卖出操作失败: {e}")
            return {
                'symbol': signal.symbol,
                'action': 'SELL',
                'success': False,
                'message': str(e),
                'order_id': None
            }

    def _execute_short(self, signal: TradingSignal, account_info: Dict) -> Dict:
        """执行做空操作"""
        try:
            # 计算交易数量
            available = account_info.get('available', 0)
            max_amount = available * self.max_position_ratio

            trade_amount = max_amount * 0.1  # 使用10%的可用资金
            price = signal.price
            volume = int(trade_amount / price / 100) * 100

            if volume <= 0:
                return {
                    'symbol': signal.symbol,
                    'action': 'SELL',
                    'success': False,
                    'message': '资金不足,无法做空',
                    'order_id': None
                }

            # 执行卖出(做空)
            order = self.tqsdk.place_order(signal.symbol, 'SELL', 'OPEN', volume)

            if order.get('order_id'):
                self.positions[signal.symbol] = {
                    'direction': 'SHORT',
                    'volume': volume,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'stop_loss': price * (1 + self.stop_loss_ratio),
                    'take_profit': price * (1 - self.take_profit_ratio)
                }

                return {
                    'symbol': signal.symbol,
                    'action': 'SELL',
                    'success': True,
                    'message': '卖出(做空)订单已提交',
                    'order_id': order['order_id'],
                    'volume': volume,
                    'price': price
                }
            else:
                return {
                    'symbol': signal.symbol,
                    'action': 'SELL',
                    'success': False,
                    'message': '卖出订单失败',
                    'order_id': None
                }

        except Exception as e:
            logger.error(f"执行做空操作失败: {e}")
            return {
                'symbol': signal.symbol,
                'action': 'SELL',
                'success': False,
                'message': str(e),
                'order_id': None
            }

    def _close_position(self, symbol: str) -> Dict:
        """平仓操作"""
        try:
            result = self.tqsdk.close_position(symbol)

            if result.get('success'):
                # 从持仓中移除
                if symbol in self.positions:
                    position = self.positions.pop(symbol)
                    profit_loss = self._calculate_profit_loss(position, symbol)

                    return {
                        'symbol': symbol,
                        'action': 'CLOSE',
                        'success': True,
                        'message': '平仓成功',
                        'order_id': None,
                        'volume': position['volume'],
                        'price': self.tqsdk.get_quote(symbol).get('last_price', 0),
                        'profit_loss': profit_loss
                    }

            return {
                'symbol': symbol,
                'action': 'CLOSE',
                'success': False,
                'message': '平仓失败',
                'order_id': None
            }

        except Exception as e:
            logger.error(f"平仓操作失败: {e}")
            return {
                'symbol': symbol,
                'action': 'CLOSE',
                'success': False,
                'message': str(e),
                'order_id': None
            }

    def _calculate_profit_loss(self, position: Dict, symbol: str) -> float:
        """计算盈亏"""
        try:
            current_price = self.tqsdk.get_quote(symbol).get('last_price', 0)
            volume = position['volume']

            if position['direction'] == 'LONG':
                profit_loss = (current_price - position['entry_price']) * volume
            else:
                profit_loss = (position['entry_price'] - current_price) * volume

            return profit_loss

        except Exception as e:
            logger.error(f"计算盈亏失败: {e}")
            return 0.0

    def _check_risk_management(self) -> List[Dict]:
        """检查风险管理"""
        risk_actions = []

        for symbol, position in self.positions.items():
            try:
                current_price = self.tqsdk.get_quote(symbol).get('last_price', 0)

                if position['direction'] == 'LONG':
                    # 多头止损检查
                    if current_price <= position['stop_loss']:
                        action = self._close_position(symbol)
                        if action['success']:
                            risk_actions.append({
                                'type': 'STOP_LOSS',
                                'symbol': symbol,
                                'action': action
                            })

                    # 多头止盈检查
                    elif current_price >= position['take_profit']:
                        action = self._close_position(symbol)
                        if action['success']:
                            risk_actions.append({
                                'type': 'TAKE_PROFIT',
                                'symbol': symbol,
                                'action': action
                            })

                else:
                    # 空头止损检查
                    if current_price >= position['stop_loss']:
                        action = self._close_position(symbol)
                        if action['success']:
                            risk_actions.append({
                                'type': 'STOP_LOSS',
                                'symbol': symbol,
                                'action': action
                            })

                    # 空头止盈检查
                    elif current_price <= position['take_profit']:
                        action = self._close_position(symbol)
                        if action['success']:
                            risk_actions.append({
                                'type': 'TAKE_PROFIT',
                                'symbol': symbol,
                                'action': action
                            })

            except Exception as e:
                logger.error(f"检查{symbol}风险管理失败: {e}")

        return risk_actions

    def _record_trade(self, signal: TradingSignal, result: Dict):
        """记录交易"""
        trade = {
            'timestamp': datetime.now(),
            'symbol': signal.symbol,
            'action': result['action'],
            'volume': result.get('volume', 0),
            'price': result.get('price', 0),
            'order_id': result.get('order_id'),
            'signal_type': signal.signal_type.value,
            'confidence': signal.confidence,
            'reasoning': signal.reasoning
        }

        self.trade_history.append(trade)
        logger.info(f"交易记录已保存: {trade}")

    def get_position_summary(self) -> str:
        """获取持仓摘要"""
        if not self.positions:
            return "暂无持仓"

        summary = f"\n=== 当前持仓 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n"

        for symbol, position in self.positions.items():
            try:
                current_price = self.tqsdk.get_quote(symbol).get('last_price', 0)
                profit_loss = self._calculate_profit_loss(position, symbol)

                summary += f"品种: {symbol}\n"
                summary += f"方向: {position['direction']}\n"
                summary += f"数量: {position['volume']}\n"
                summary += f"开仓价: {position['entry_price']:.2f}\n"
                summary += f"当前价: {current_price:.2f}\n"
                summary += f"浮动盈亏: {profit_loss:.2f}\n"
                summary += f"止损价: {position['stop_loss']:.2f}\n"
                summary += f"止盈价: {position['take_profit']:.2f}\n\n"
            except Exception as e:
                summary += f"获取{symbol}信息失败: {e}\n\n"

        return summary

    def get_trade_history(self, limit: int = 10) -> List[Dict]:
        """获取交易历史"""
        return self.trade_history[-limit:]

    def start_trading(self, signals: List[TradingSignal]) -> Dict:
        """开始执行交易"""
        self.is_running = True
        results = []
        risk_actions = []
        total_profit = 0

        logger.info(f"开始执行交易,共{len(signals)}个信号")

        # 先获取初始账户信息
        initial_account = self.tqsdk.get_account_info()
        initial_balance = initial_account.get('balance', 0)

        for signal in signals:
            if not self.is_running:
                break

            logger.info(f"执行信号: {signal.symbol} {signal.signal_type.value}")

            # 检查是否已经在交易这个品种
            if signal.symbol in self.positions:
                logger.info(f"{signal.symbol} 已有持仓,跳过")
                continue

            # 执行信号
            account_info = self.tqsdk.get_account_info()
            result = self.execute_signal(signal, account_info)
            results.append(result)

            # 记录交易
            if result['success']:
                logger.info(f"✅ {signal.symbol} 交易执行成功")
            else:
                logger.warning(f"❌ {signal.symbol} 交易执行失败: {result['message']}")

            # 检查风险管理
            new_risk_actions = self._check_risk_management()
            risk_actions.extend(new_risk_actions)

            # 短暂等待,避免交易过于频繁
            time.sleep(0.5)

        # 更新账户信息
        final_account = self.tqsdk.get_account_info()
        final_balance = final_account.get('balance', 0)
        total_profit = final_balance - initial_balance

        self.is_running = False

        # 生成交易总结
        trade_summary = self._generate_trade_summary(signals, results, total_profit)

        return {
            'results': results,
            'risk_actions': risk_actions,
            'total_profit': total_profit,
            'trade_summary': trade_summary,
            'initial_balance': initial_balance,
            'final_balance': final_balance
        }

    def stop_trading(self):
        """停止交易"""
        self.is_running = False
        logger.info("交易已停止")

    def _generate_trade_summary(self, signals: List[TradingSignal], results: List[Dict], total_profit: float) -> str:
        """生成交易总结"""
        summary = f"\n=== 交易总结 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n"

        # 统计结果
        successful_trades = sum(1 for r in results if r['success'])
        failed_trades = len(results) - successful_trades

        summary += f"交易统计:\n"
        summary += f"• 总交易数: {len(results)}\n"
        summary += f"• 成功交易: {successful_trades}\n"
        summary += f"• 失败交易: {failed_trades}\n"
        summary += f"• 成功率: {successful_trades/len(results)*100:.1f}%\n\n"

        # 盈亏统计
        summary += f"盈亏统计:\n"
        summary += f"• 总盈亏: {total_profit:.2f}\n"
        if total_profit > 0:
            summary += "• 盈利状态\n"
        elif total_profit < 0:
            summary += "• 亏损状态\n"
        else:
            summary += "• 平衡状态\n\n"

        # 风险管理统计
        summary += f"风险管理:\n"
        active_positions = len(self.positions)
        summary += f"• 当前持仓: {active_positions}\n"

        if self.positions:
            summary += f"• 持仓详情:\n"
            for symbol, position in self.positions.items():
                current_price = self.tqsdk.get_quote(symbol).get('last_price', 0)
                profit_loss = self._calculate_profit_loss(position, symbol)
                summary += f"  - {symbol}: {position['direction']} {position['volume']}手, 浮动盈亏: {profit_loss:.2f}\n"

        # 建议
        summary += f"\n交易建议:\n"
        if failed_trades > successful_trades:
            summary += "• 建议检查信号质量,可能需要调整策略\n"
        elif active_positions > 0:
            summary += "• 建议密切关注持仓风险,及时止损\n"
        else:
            summary += "• 当前无持仓,可以等待新的交易机会\n"

        return summary

    def get_portfolio_summary(self) -> str:
        """获取投资组合摘要"""
        summary = f"\n=== 投资组合摘要 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n"

        # 获取账户信息
        account_info = self.tqsdk.get_account_info()

        summary += f"账户信息:\n"
        summary += f"• 账户余额: {account_info.get('balance', 0):.2f}\n"
        summary += f"• 可用资金: {account_info.get('available', 0):.2f}\n"
        summary += f"• 持仓保证金: {account_info.get('margin', 0):.2f}\n"
        summary += f"• 浮动盈亏: {account_info.get('float_profit', 0):.2f}\n"
        summary += f"• 风险度: {account_info.get('risk_ratio', 0):.2%}\n\n"

        # 持仓信息
        if self.positions:
            summary += f"当前持仓:\n"
            total_position_value = 0

            for symbol, position in self.positions.items():
                current_price = self.tqsdk.get_quote(symbol).get('last_price', 0)
                position_value = position['volume'] * current_price
                profit_loss = self._calculate_profit_loss(position, symbol)
                total_position_value += position_value

                summary += f"• {symbol}:\n"
                summary += f"  - 方向: {position['direction']}\n"
                summary += f"  - 数量: {position['volume']}手\n"
                summary += f"  - 开仓价: {position['entry_price']:.2f}\n"
                summary += f"  - 当前价: {current_price:.2f}\n"
                summary += f"  - 浮动盈亏: {profit_loss:.2f}\n"
                summary += f"  - 止损价: {position['stop_loss']:.2f}\n"
                summary += f"  - 止盈价: {position['take_profit']:.2f}\n"
        else:
            summary += "当前无持仓\n"

        # 风险提示
        risk_ratio = account_info.get('risk_ratio', 0)
        if risk_ratio > 0.8:
            summary += f"\n⚠️  风险警告: 当前风险度{risk_ratio:.2%}过高,建议降低仓位\n"
        elif risk_ratio > 0.6:
            summary += f"\n⚠️  注意事项: 当前风险度{risk_ratio:.2%}较高,请谨慎操作\n"

        return summary

    def close_all_positions(self) -> Dict:
        """平掉所有持仓"""
        results = []

        for symbol in list(self.positions.keys()):
            result = self._close_position(symbol)
            results.append({
                'symbol': symbol,
                'success': result['success'],
                'message': result.get('message', '')
            })

        return {
            'action': 'close_all_positions',
            'results': results,
            'total_closed': len(results)
        }

    def get_risk_metrics(self) -> Dict:
        """获取风险指标"""
        account_info = self.tqsdk.get_account_info()

        # 计算各项风险指标
        risk_metrics = {
            'risk_ratio': account_info.get('risk_ratio', 0),
            'margin_ratio': account_info.get('margin', 0) / max(account_info.get('balance', 1), 1),
            'position_count': len(self.positions),
            'total_float_profit': account_info.get('float_profit', 0),
            'total_margin': account_info.get('margin', 0)
        }

        # 计算持仓风险
        position_risks = []
        for symbol, position in self.positions.items():
            current_price = self.tqsdk.get_quote(symbol).get('last_price', 0)
            profit_loss = self._calculate_profit_loss(position, symbol)
            risk_percentage = abs(profit_loss) / position['volume'] / position['entry_price']

            position_risks.append({
                'symbol': symbol,
                'profit_loss': profit_loss,
                'risk_percentage': risk_percentage
            })

        risk_metrics['position_risks'] = position_risks

        return risk_metrics

    def close(self):
        """关闭连接"""
        self.tqsdk.close()