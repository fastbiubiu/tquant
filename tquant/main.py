"""
量化交易 Agent 系统主程序 - 第一周MVP版本
"""

import logging
import signal
import sys
import time
from datetime import datetime

from tquant.agents.market_analyst import MarketAnalyst
from tquant.agents.trader import Trader
from tquant.config import get_config, Config
from tquant.utils.signals import SignalType
from tquant.utils.tqsdk_interface import TqSdkInterface

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class TradingSystem:
    """量化交易系统主类 - 第一周MVP版本"""

    def __init__(self, config_path: str = None):
        """
        初始化交易系统

        Args:
            config_path: 配置文件路径(可选,使用默认配置)
        """
        logger.info("=" * 50)
        logger.info("量化交易 Agent 系统启动 - 第一周MVP版本")
        logger.info("=" * 50)

        # 加载配置(使用新的 Pydantic 配置系统)
        self.config: Config = get_config()
        self._setup_logging()

        # 初始化 tqsdk 接口
        self.api = TqSdkInterface(config_path)

        # 初始化 Agent
        self.market_analyst = MarketAnalyst(config_path)
        self.trader = Trader(config_path)

        # 初始化 LLM(第三阶段)
        self.enable_llm = self.config.llm.gpt4o is not None if self.config.llm else False

        # 系统状态
        self.is_running = False
        self.symbols_to_analyze = self.config.trading.symbols

        # 添加信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("系统初始化完成")

    def _setup_logging(self):
        """设置日志格式"""
        log_level = self.config.logging.level if self.config.logging else 'INFO'
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(numeric_level)

    def _initialize_api(self) -> TqSdkInterface:
        """
        初始化 tqsdk 接口

        Returns:
            tqsdk 接口实例
        """
        try:
            system_config = self.config.system.model_dump() if hasattr(self.config, 'system') else {}

            api = TqSdkInterface(
                account=system_config.get('account') or self.config.tqsdk.auth,
                password=system_config.get('password') or self.config.tqsdk.auth,
                debug=system_config.get('debug_mode', False) or self.config.tqsdk.demo
            )

            logger.info("tqsdk 接口初始化成功")
            return api

        except Exception as e:
            logger.error(f"tqsdk 接口初始化失败: {e}")
            raise

    def _initialize_market_analyst(self) -> MarketAnalyst:
        """
        初始化市场分析师

        Returns:
            市场分析师实例
        """
        try:
            analyst = MarketAnalyst(
                api=self.api,
                config=self.config,
                enable_llm=self.enable_llm
            )

            logger.info("市场分析师初始化成功")
            return analyst

        except Exception as e:
            logger.error(f"市场分析师初始化失败: {e}")
            raise


    def _signal_handler(self, signum, frame):
        """信号处理"""
        logger.info(f"收到信号 {signum},正在关闭系统...")
        self.stop()

    def start(self):
        """启动交易系统"""
        logger.info("启动交易系统...")
        self.is_running = True

        try:
            # 获取刷新间隔
            refresh_interval = self.config.system.refresh_interval if hasattr(self.config, 'system') else 60

            while self.is_running:
                start_time = time.time()

                try:
                    # 1. 获取当前报价
                    quote = self.api.get_quote(self.symbols_to_analyze[0])
                    current_price = quote['last_price']

                    logger.info(f"\n{'=' * 60}")
                    logger.info(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info(f"合约: {self.symbols_to_analyze[0]}")
                    logger.info(f"价格: {current_price:.2f}")
                    logger.info(f"{'=' * 60}\n")

                    # 2. 分析市场
                    signal = self.market_analyst.analyze_symbol(
                        self.symbols_to_analyze[0],
                        interval=300  # 5分钟
                    )

                    if signal:
                        logger.info(f"分析结果: {signal.signal_type.value} (置信度: {signal.confidence:.2f})")
                        logger.info(f"理由: {signal.reason}")
                    else:
                        logger.info("分析失败,跳过本次交易")
                        time.sleep(refresh_interval)
                        continue

                    # 3. 执行交易(如果信号不是 HOLD)
                    if signal.signal_type != SignalType.HOLD:
                        logger.info(f"执行交易: {signal.signal_type.value}")

                        # 风险检查
                        risk_check = self.trader.pre_trade_check(
                            signal,
                            current_price
                        )

                        if risk_check['passed']:
                            # 执行交易
                            trade_result = self.trader.execute(signal, current_price)

                            if trade_result and trade_result['success']:
                                logger.info(f"交易执行成功: {trade_result}")

                                # 记录持仓
                                self.trader.post_trade_check(signal, current_price)
                            else:
                                logger.warning(f"交易执行失败: {trade_result}")

                                # 记录风险检查失败
                                for check in risk_check['checks']:
                                    if not check['passed']:
                                        logger.warning(f"风险检查失败: {check['reason']}")
                        else:
                            logger.warning(f"交易被拒绝: {risk_check['reason']}")
                    else:
                        logger.info("无交易信号,保持持仓")

                    # 4. 监控持仓
                    self.trader.position_monitor()

                    # 5. 打印统计
                    self._print_statistics()

                    # 6. 等待下一次刷新
                    elapsed = time.time() - start_time
                    sleep_time = max(1, refresh_interval - int(elapsed))
                    logger.info(f"等待 {sleep_time} 秒后进行下一次分析...")
                    time.sleep(sleep_time)

                except Exception as e:
                    logger.error(f"交易过程中出错: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(refresh_interval)

        except KeyboardInterrupt:
            logger.info("用户中断")
        except Exception as e:
            logger.error(f"系统运行错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """停止交易系统"""
        logger.info("正在停止系统...")
        self.is_running = False

        try:
            # 关闭 API 连接
            self.api.close()
            logger.info("API 连接已关闭")

            # 打印最终统计
            self._print_statistics()

            logger.info("系统已停止")

        except Exception as e:
            logger.error(f"停止系统时出错: {e}")

    def _print_statistics(self):
        """打印统计信息"""
        logger.info("\n" + "=" * 60)
        logger.info("系统统计信息")
        logger.info("=" * 60)

        # 市场分析统计
        analyst_stats = self.market_analyst.get_analysis_statistics()
        logger.info(f"市场分析统计:")
        logger.info(f"  总分析次数: {analyst_stats['total_analyses']}")
        logger.info(f"  已分析合约数: {analyst_stats['analyzed_symbols_count']}")
        logger.info(f"  启用 LLM: {analyst_stats['enable_llm']}")

        # 信号统计
        signal_stats = self.market_analyst.signal_generator.get_signal_statistics()
        logger.info(f"\n信号统计:")
        logger.info(f"  总信号数: {signal_stats['total_signals']}")
        logger.info(f"  买入信号: {signal_stats['buy_signals']}")
        logger.info(f"  卖出信号: {signal_stats['sell_signals']}")
        logger.info(f"  持有信号: {signal_stats['hold_signals']}")
        logger.info(f"  买入比例: {signal_stats['buy_ratio']:.1%}")
        logger.info(f"  卖出比例: {signal_stats['sell_ratio']:.1%}")
        logger.info(f"  平均置信度: {signal_stats['avg_confidence']:.2f}")

        # 交易统计
        trade_stats = self.trader.get_trades_statistics()
        logger.info(f"\n交易统计:")
        logger.info(f"  总交易数: {trade_stats['total_trades']}")
        logger.info(f"  成功交易: {trade_stats['successful_trades']}")
        logger.info(f"  失败交易: {trade_stats['failed_trades']}")
        logger.info(f"  胜率: {trade_stats['win_rate']:.1%}")
        logger.info(f"  总收益: {trade_stats['total_profit']:.2f}")
        logger.info(f"  平均收益: {trade_stats['avg_profit']:.2f}")
        logger.info(f"  每日交易数: {trade_stats['daily_trades']}")
        logger.info(f"  当前持仓数: {trade_stats['current_positions']}")

        # 风险管理统计
        risk_stats = self.trader.get_risk_statistics()
        logger.info(f"\n风险管理统计:")
        logger.info(f"  风险检查总数: {risk_stats['total_checks']}")
        logger.info(f"  通过检查: {risk_stats['passed_checks']}")
        logger.info(f"  失败检查: {risk_stats['failed_checks']}")
        logger.info(f"  通过率: {risk_stats['pass_rate']:.1%}")
        logger.info(f"  止损触发: {risk_stats['occurred_stop_losses']}")
        logger.info(f"  止盈触发: {risk_stats['occurred_take_profits']}")
        logger.info(f"  最大持仓数: {risk_stats['current_positions']}")

        logger.info("=" * 60 + "\n")


def main():
    """主函数"""
    try:
        # 创建并启动系统
        system = TradingSystem('config.yaml')
        system.start()

    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
