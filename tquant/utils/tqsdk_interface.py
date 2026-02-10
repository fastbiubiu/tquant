"""
tqsdk接口封装模块
提供与天勤量化API的交互接口
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqsdk import TqApi, TqBacktest, TqAccount
from tqsdk.auth import TqAuth
from tqsdk.objs import Quote

import tquant.config.schema
from tquant.config import get_config, Config

logger = logging.getLogger(__name__)


class TqSdkInterface:
    """tqsdk API接口封装类"""

    def __init__(self):
        """初始化tqsdk接口"""
        self.config: tquant.config.schema.TQSDKConfig = get_config().tqsdk
        self.api: Optional[TqApi] = None
        self.quote: Optional[Quote] = None
        self.klines: Optional[pd.DataFrame] = None

    def connect(self, backtest: tquant.config.schema.BacktestConfig = None,
                account: tquant.config.schema.AccountConfig = None, symbol: str = "") -> bool:
        """连接tqsdk API

        Args:

        Returns:
            bool: 连接是否成功
        """
        # 已连接则直接复用
        if self.api is not None:
            logger.info("tqsdk API 已连接, 复用现有连接")
            return True
        backtest = backtest if backtest is not None else self.config.backtest
        account = account if account is not None else self.config.account
        backtest_obj = None
        account_obj = None
        if backtest is not None:
            backtest_obj = TqBacktest(start_dt=backtest.start_date, end_dt=backtest.end_date)
        if account is not None:
            account_obj = TqAccount(account.broker_id, account.account_id, account.password)
        try:
            auth_obj = TqAuth(self.config.auth.username, self.config.auth.password)
            self.api = TqApi(
                account=account_obj,
                backtest=backtest_obj,
                auth=auth_obj,
            )
            symbol = symbol if symbol else self.config.symbol
            self.quote = self.api.get_quote(symbol)
            self.klines = self.api.get_kline_serial(symbol, 60)
            logger.info("✅ tqsdk API连接成功")
            return True
        except Exception as e:
            logger.error(f"❌ tqsdk API连接失败: {e}")
            logger.error("提示: 请确保已在配置文件或环境变量中设置有效的 tqsdk 认证信息")
            logger.error("1. 配置文件路径: ~/.tquant/config.json")
            logger.error("3. 注册地址: https://account.shinnytech.com/")
            import traceback
            traceback.print_exc()
            self.api = None
            return False

    def get_quote(self, symbol: str) -> Dict:
        """获取实时行情数据"""
        quote_data = {
            'symbol': symbol,
            'last_price': self.quote.last_price,
            'volume': self.quote.volume,
            'open': self.quote.open,
            'high': self.quote.highest,
            'low': self.quote.lowest,
            'close': self.quote.close,
            'datetime': datetime.now().isoformat()
        }

        return quote_data

    def get_kline_data(self) -> pd.DataFrame:
        """获取K线数据"""

        # 转换为DataFrame
        data = []
        for i in range(len(self.klines)):
            kline = self.klines.iloc[i]
            data.append({
                'datetime': kline.datetime,
                'open': kline.open,
                'high': kline.high,
                'low': kline.low,
                'close': kline.close,
                'volume': kline.volume,
            })

        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)

        return df

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

    def wait_update(self) -> bool:
        """等待数据更新"""
        if self.api:
            self.api.wait_update()
            if self.api.is_changing(self.quote, "last_price"):
                return True
        return False

    def close(self):
        """关闭连接"""
        if self.api:
            self.api.close()
            self.api = None
            logger.info("tqsdk API连接已关闭")
