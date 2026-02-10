"""
量化交易 Agent 系统主程序 - 第一周MVP版本
"""

import asyncio
import logging
import os
import signal
import sys
import json
from tqsdk import TqApi, TqAuth, TqAccount
from tqsdk.ta import MA
from tquant.config import loader as config_loader
from tquant.strategy import MyTradingStrategy # 导入您的策略类

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

async def start():
    logger.info("Starting Tquant trading system...")

    # 1. 加载配置
    config_file = os.path.join(os.path.expanduser("~"), ".tquant", "config.json")
    config = config_loader.load_config(config_file)
    logger.info(f"Loading configuration from {config_file}")

    # 2. 从配置中获取交易合约和账户信息
    trading_instrument = config.get("trading_instrument", "SHFE.cu2401") # 默认合约
    account_id = config.get("account_id", "sim_account") # 默认模拟账户
    tq_username = config.get("tq_username")
    tq_password = config.get("tq_password")

    if not tq_username or not tq_password:
        logger.error("天勤用户名或密码未配置。请检查 config.json 文件。")
        sys.exit(1)

    # 3. 初始化天勤SDK API
    logger.info("Connecting to tqsdk API...")
    try:
        api = TqApi(
            TqAccount(account_id, tq_username, tq_password),
            auth=TqAuth(tq_username, tq_password)
        )
        logger.info("tqsdk API 连接成功。")
    except Exception as e:
        logger.critical(f"连接 tqsdk API 失败: {e}")
        sys.exit(1)

    # 4. 初始化策略
    strategy = MyTradingStrategy(api, account_id)
    await strategy.initialize(trading_instrument)

    # 5. 订阅行情
    logger.info(f"订阅合约行情: {trading_instrument}")
    quote = await api.get_quote(trading_instrument)
    if not quote:
        logger.error(f"获取 {trading_instrument} 行情数据失败: quote 对象为空。")
        await api.close()
        sys.exit(1)

    # 6. 主循环：持续获取行情并传递给策略
    try:
        while True:
            await api.wait_update() # 等待数据更新

            # 检查行情数据是否有效
            if quote.datetime == "0": # Tqsdk 在盘前或无数据时 datetime 可能为 "0"
                # print(f"等待 {trading_instrument} 有效行情数据...")
                continue

            # 将最新的行情数据传递给策略处理
            await strategy.on_quote(quote)

            # 您可以在这里添加其他系统级的监控或管理逻辑
            await strategy.run_strategy_loop() # 如果策略有自己的周期性任务

    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("交易系统被用户中断。")
    except Exception as e:
        logger.critical(f"交易过程中出错: {e}", exc_info=True)
    finally:
        logger.info("关闭天勤API和策略...")
        await strategy.close()
        api.close()
        logger.info("交易系统已停止。")


def stop_handler(signum, frame):
    logger.info(f"接收到停止信号 {signum}，正在退出...")
    sys.exit(0) # 退出主程序

def main():
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    try:
        asyncio.run(start())
    except SystemExit as e:
        logger.info(f"系统正常退出: {e}")
    except Exception as e:
        logger.critical(f"主程序异常退出: {e}", exc_info=True)

if __name__ == "__main__":
    main()
