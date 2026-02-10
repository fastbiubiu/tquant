"""
tqsdk接口封装模块
提供与天勤量化API的交互接口
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import tqsdk
from tqsdk import TqBacktest, TqKq, TqKqStock
from tqsdk.auth import TqAuth

from tquant.config import get_config, Config

logger = logging.getLogger(__name__)


class TqSdkInterface:
    """tqsdk API接口封装类"""

    def __init__(self, config_path: str = None):
        """初始化tqsdk接口"""
        self.config: Config = get_config()
        self.api = None
        self.quote_dict = {}

    def connect(self, backtest: bool = None, demo: bool = None):
        """连接tqsdk API

        Args:
            backtest: 是否启用回测模式（True=回测，False=实盘）
            demo: 是否启用模拟盘（仅实盘模式有效，True=模拟盘，False=实盘）

        Returns:
            bool: 连接是否成功
        """
        try:
            # 从配置文件获取值
            self.config: Config = get_config()
            backtest = backtest if backtest is not None else (self.config.tqsdk.backtest if self.config.tqsdk else False)
            demo = demo if demo is not None else (self.config.tqsdk.demo if self.config.tqsdk else True)

            # 组装 auth 对象（必填参数）
            if backtest:
                # 回测模式：使用模拟账户
                if demo:
                    # 模拟盘不需要 auth，但 tqsdk 要求提供
                    auth = tqsdk.auth.TqAuth('demo', 'demo')
                    logger.info("✅ 启用回测模式（模拟账户）")
                else:
                    # 实盘回测需要 auth
                    if self.config.tqsdk and self.config.tqsdk.auth and isinstance(self.config.tqsdk.auth, str):
                        auth_str = self.config.tqsdk.auth
                        if ':' in auth_str:
                            parts = auth_str.split(':')
                            if len(parts) >= 2:
                                auth = tqsdk.auth.TqAuth(parts[0], parts[1])
                            else:
                                auth = tqsdk.auth.TqAuth(parts[0], parts[0])
                        else:
                            auth = tqsdk.auth.TqAuth(auth_str, auth_str)
                    else:
                        auth = tqsdk.auth.TqAuth('demo', 'demo')
            else:
                # 实盘模式：根据是否有配置决定
                if self.config.tqsdk and self.config.tqsdk.auth and isinstance(self.config.tqsdk.auth, str):
                    auth_str = self.config.tqsdk.auth
                    if ':' in auth_str:
                        parts = auth_str.split(':')
                        if len(parts) >= 2:
                            auth = tqsdk.auth.TqAuth(parts[0], parts[1])
                        else:
                            auth = tqsdk.auth.TqAuth(parts[0], parts[0])
                    else:
                        auth = tqsdk.auth.TqAuth(auth_str, auth_str)
                else:
                    raise ValueError("实盘模式需要配置 TQUANT_TQSDK_AUTH 环境变量或 config.tqsdk.auth")

            # 组装 account 对象
            if backtest:
                # 回测模式：使用 TqSim 或 TqSimStock
                account = tqsdk.tradeable.sim.tqsim.TqSim() if demo else tqsdk.tradeable.sim.tqsim_stock.TqSimStock()
                logger.info("✅ 启用回测模式")
            else:
                # 实盘模式：根据 demo 参数选择 TqKq 或 TqKqStock
                if demo:
                    account = tqsdk.tradeable.otg.tqkq.TqKq()
                    logger.info("✅ 使用模拟盘模式")
                else:
                    account = tqsdk.tradeable.otg.tqkq.TqKqStock()
                    logger.info("✅ 使用实盘模式")

            # 组装 backtest 对象
            if backtest:
                backtest_obj = tqsdk.backtest.backtest.TqBacktest(start_dt=datetime(2020, 1, 1), end_dt=datetime(2023, 12, 31))
            else:
                backtest_obj = None

            # 创建 TqApi 实例
            self.api = tqsdk.TqApi(
                account=account,
                auth=auth,
                backtest=backtest_obj
            )

            logger.info("✅ tqsdk API连接成功")
            return True

        except Exception as e:
            logger.error(f"❌ tqsdk API连接失败: {e}")
            logger.error(f"\n提示: 请确保已配置有效的 tqsdk 认证信息")
            logger.error(f"1. 在 ~/.tquant/config.json 中设置 tqsdk.auth 或使用环境变量 TQUANT_TQSDK_AUTH")
            logger.error(f"2. 格式: account:password")
            logger.error(f"3. 注册地址: https://account.shinnytech.com/")
            import traceback
            traceback.print_exc()
            return False

    def get_quote(self, symbol: str) -> Dict:
        """获取实时行情数据"""
        try:
            if not self.api:
                raise Exception("API未连接")

            quote = self.api.get_quote(symbol)
            # 等待数据更新
            self.api.wait_update()

            quote_data = {
                'symbol': symbol,
                'last_price': quote.last_price,
                'volume': quote.volume,
                'open': quote.open,
                'high': quote.high,
                'low': quote.low,
                'close': quote.close,
                'datetime': datetime.now().isoformat()
            }

            return quote_data

        except Exception as e:
            logger.error(f"获取{symbol}行情数据失败: {e}")
            return {}

    def get_kline_data(self, symbol: str, period: str = "1m", count: int = 100) -> pd.DataFrame:
        """获取K线数据"""
        try:
            if not self.api:
                raise Exception("API未连接")

            # 获取K线数据
            klines = self.api.get_kline_serial(symbol, period, count)

            # 等待数据更新
            self.api.wait_update()

            # 转换为DataFrame
            data = []
            for i in range(len(klines)):
                kline = klines.iloc[i]
                data.append({
                    'datetime': kline.datetime,
                    'open': kline.open,
                    'high': kline.high,
                    'low': kline.low,
                    'close': kline.close,
                    'volume': kline.volume,
                    'symbol': symbol
                })

            df = pd.DataFrame(data)
            df.set_index('datetime', inplace=True)

            return df

        except Exception as e:
            logger.error(f"获取{symbol} K线数据失败: {e}")
            return pd.DataFrame()

    def get_account_info(self) -> Dict:
        """获取账户信息"""
        try:
            if not self.api:
                raise Exception("API未连接")

            account = self.api.get_account()
            self.api.wait_update()

            return {
                'balance': account.balance,
                'available': account.available,
                'position_profit': account.position_profit,
                'float_profit': account.float_profit,
                'margin': account.margin,
                'risk_ratio': account.risk_ratio
            }

        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
            return {}

    def place_order(self, symbol: str, direction: str, offset: str, volume: int) -> Dict:
        """下单"""
        try:
            if not self.api:
                raise Exception("API未连接")

            # 创建订单
            order = self.api.insert_order(
                symbol=symbol,
                direction=direction,
                offset=offset,
                volume=volume,
                limit_price=None,  # 市价单
                order_type='ANY'
            )

            logger.info(f"下单成功: {symbol} {direction} {volume}")

            return {
                'order_id': order.order_id,
                'symbol': symbol,
                'direction': direction,
                'offset': offset,
                'volume': volume,
                'status': 'PENDING'
            }

        except Exception as e:
            logger.error(f"下单失败: {e}")
            return {}

    def close_position(self, symbol: str) -> Dict:
        """平仓"""
        try:
            if not self.api:
                raise Exception("API未连接")

            # 获取当前持仓
            position = self.api.get_position(symbol)

            if position.pos_long > 0:
                # 多头平仓
                order = self.api.insert_order(
                    symbol=symbol,
                    direction='SELL',
                    offset='CLOSE',
                    volume=int(position.pos_long),
                    limit_price=None,
                    order_type='ANY'
                )
                logger.info(f"平多头仓位: {symbol} {position.pos_long}")

            elif position.pos_short > 0:
                # 空头平仓
                order = self.api.insert_order(
                    symbol=symbol,
                    direction='BUY',
                    offset='CLOSE',
                    volume=int(position.pos_short),
                    limit_price=None,
                    order_type='ANY'
                )
                logger.info(f"平空头仓位: {symbol} {position.pos_short}")

            return {
                'symbol': symbol,
                'success': True
            }

        except Exception as e:
            logger.error(f"平仓失败: {e}")
            return {'symbol': symbol, 'success': False, 'error': str(e)}

    def subscribe(self, symbols: List[str]):
        """订阅行情"""
        try:
            for symbol in symbols:
                self.quote_dict[symbol] = self.api.get_quote(symbol)
            logger.info(f"订阅成功: {symbols}")
        except Exception as e:
            logger.error(f"订阅失败: {e}")

    def wait_update(self):
        """等待数据更新"""
        if self.api:
            self.api.wait_update()

    def close(self):
        """关闭连接"""
        if self.api:
            self.api.close()
            logger.info("tqsdk API连接已关闭")