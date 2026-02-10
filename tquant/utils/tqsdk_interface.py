"""
tqsdk接口封装模块
提供与天勤量化API的交互接口
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqsdk import TqApi, TqBacktest, TqSim
from tqsdk.auth import TqAuth

from tquant.config import get_config, Config

logger = logging.getLogger(__name__)


class TqSdkInterface:
    """tqsdk API接口封装类"""

    def __init__(self, config_path: Optional[str] = None):
        """初始化tqsdk接口"""
        # 如果提供了 config_path，则优先从指定配置文件加载，否则使用默认全局配置
        self.config: Config = get_config(Path(config_path)) if config_path else get_config()
        self.api: Optional[TqApi] = None
        self.quote_dict: Dict[str, object] = {}

    def _parse_auth(self, auth_str: str) -> Optional[TqAuth]:
        """
        从配置中的认证字符串构造 TqAuth。
        支持两种格式：
        - "username,password"
        - "username:password"
        """
        if not auth_str:
            return None
        try:
            sep = "," if "," in auth_str else ":"
            if sep in auth_str:
                user, pwd = auth_str.split(sep, 1)
            else:
                user = pwd = auth_str
            return TqAuth(user.strip(), pwd.strip())
        except Exception as e:
            logger.error(f"解析 TQSDK 认证信息失败: {e}")
            return None

    def connect(self, *, backtest: Optional[bool] = None, demo: Optional[bool] = None) -> bool:
        """连接tqsdk API

        Args:
            backtest: 是否启用回测模式（优先级高于配置，默认使用 config.tqsdk.backtest）
            demo: 是否启用模拟/本地模拟账户（默认使用 config.tqsdk.demo）

        Returns:
            bool: 连接是否成功
        """
        # 已连接则直接复用
        if self.api is not None:
            logger.info("tqsdk API 已连接, 复用现有连接")
            return True

        try:
            # 刷新配置，确保使用最新值
            self.config = get_config()
            tqsdk_cfg = self.config.tqsdk if self.config and self.config.tqsdk else None

            backtest_flag = backtest if backtest is not None else (tqsdk_cfg.backtest if tqsdk_cfg else False)
            demo_flag = demo if demo is not None else (tqsdk_cfg.demo if tqsdk_cfg else True)

            # 认证信息：从 config.tqsdk.auth 解析，支持逗号或冒号分隔
            auth_obj: Optional[TqAuth] = self._parse_auth(tqsdk_cfg.auth) if tqsdk_cfg and tqsdk_cfg.auth else None

            # 账户与回测配置
            account = None
            backtest_obj = None

            if backtest_flag:
                # 回测：使用 TqSim 账户和配置文件中的回测时间区间/初始资金
                bt_cfg = getattr(self.config, "backtest", None)
                initial_balance = bt_cfg.initial_balance if bt_cfg else 1_000_000.0
                account = TqSim(initial_balance)

                if bt_cfg:
                    backtest_obj = TqBacktest(start_dt=bt_cfg.start_date, end_dt=bt_cfg.end_date)
                else:
                    # 没有 backtest 段时给一个相对安全的默认区间
                    backtest_obj = TqBacktest(start_dt=datetime(2020, 1, 1), end_dt=datetime(2023, 12, 31))

                logger.info("✅ 启用回测模式 (TqSim + TqBacktest)")
            else:
                # 非回测：根据 demo 决定是否使用本地模拟账户
                if demo_flag:
                    account = TqSim()
                    logger.info("✅ 使用本地模拟账户 (TqSim)")
                else:
                    account = None  # 实盘：不传 account，由 tqsdk 根据 auth 绑定实盘账户
                    logger.info("✅ 使用实盘模式 (使用 auth 绑定账户)")

            # 创建 TqApi 实例（与官方示例一致：account + backtest + auth）
            self.api = TqApi(
                account=account,
                backtest=backtest_obj,
                auth=auth_obj,
            )

            logger.info("✅ tqsdk API连接成功")
            return True

        except Exception as e:
            logger.error(f"❌ tqsdk API连接失败: {e}")
            logger.error("提示: 请确保已在配置文件或环境变量中设置有效的 tqsdk 认证信息")
            logger.error("1. 配置文件路径: ~/.tquant/config.json 或项目 config.yaml")
            logger.error("2. 环境变量: TQUANT_TQSDK_AUTH 或 TQSDK_AUTH (格式: username,password)")
            logger.error("3. 注册地址: https://account.shinnytech.com/")
            import traceback
            traceback.print_exc()
            self.api = None
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
            if not self.api:
                raise Exception("API未连接")

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
            self.api = None
            logger.info("tqsdk API连接已关闭")