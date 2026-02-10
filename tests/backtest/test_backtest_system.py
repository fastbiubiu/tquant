"""
回测系统测试脚本
验证所有功能是否正常工作
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_tqsdk_connection():
    """测试 tqsdk 连接"""
    logger.info("="*60)
    logger.info("📋 测试 1: tqsdk 连接")
    logger.info("="*60)

    try:
        from tquant.utils.tqsdk_interface import TqSdkInterface

        # 创建接口
        interface = TqSdkInterface()

        # 连接（回测模式）
        # 回测模式会从 config 中读取 auth
        success = interface.connect(backtest=True, demo=True)

        if success:
            logger.info("✅ tqsdk 连接成功")

            # 测试获取数据
            try:
                # 获取K线数据
                df = interface.get_kline_data('SHFE.rb', period='1d', count=10)
                if not df.empty:
                    logger.info(f"✅ 获取K线数据成功: {len(df)} 天")
                    logger.info(f"  数据示例: {df.tail(3)}")
                else:
                    logger.warning("⚠️ 获取K线数据为空")
            except Exception as e:
                logger.error(f"❌ 获取K线数据失败: {e}")

            # 关闭连接
            interface.close()
        else:
            logger.error("❌ tqsdk 连接失败")

        return success

    except Exception as e:
        logger.error(f"❌ tqsdk 连接测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backtest_engine():
    """测试回测引擎"""
    logger.info("\n" + "="*60)
    logger.info("📋 测试 2: 回测引擎")
    logger.info("="*60)

    try:
        from tests.backtest.backtest_engine import BacktestEngine
        from tests.backtest.example_strategies import ma_crossover_strategy

        # 创建引擎
        engine = BacktestEngine(
            strategy_func=ma_crossover_strategy,
            initial_balance=100000.0,
            start_dt=datetime(2020, 1, 1),
            end_dt=datetime(2020, 6, 30)  # 只测试3个月
        )

        logger.info(f"✅ 回测引擎创建成功")
        logger.info(f"   策略: MA交叉策略")
        logger.info(f"   初始资金: {engine.initial_balance:,.2f}")
        logger.info(f"   时间范围: {engine.start_dt} 到 {engine.end_dt}")

        # 运行回测
        result = engine.run('SHFE.rb', period='1d')

        if result:
            logger.info("\n✅ 回测执行成功")
            logger.info(f"   最终权益: {result.final_balance:,.2f}")
            logger.info(f"   总收益率: {result.total_return:.2f}%")
            logger.info(f"   总交易次数: {result.total_trades}")
            logger.info(f"   胜率: {result.win_rate:.2%}")
            logger.info(f"   最大回撤: {result.max_drawdown:.2f}%")
            logger.info(f"   夏普比率: {result.sharpe_ratio:.2f}")
            logger.info(f"   年化收益: {result.annual_return:.2f}%")

            return True
        else:
            logger.error("❌ 回测执行失败")
            return False

    except Exception as e:
        logger.error(f"❌ 回测引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backtest_validator():
    """测试回测验证器"""
    logger.info("\n" + "="*60)
    logger.info("📋 测试 3: 回测验证器")
    logger.info("="*60)

    try:
        from tests.backtest.backtest_validator import BacktestValidator
        from tests.backtest.backtest_engine import BacktestResult, BacktestEngine
        from tests.backtest.example_strategies import rsi_strategy

        # 创建验证器
        validator = BacktestValidator()

        logger.info("✅ 回测验证器创建成功")
        logger.info(f"   验证阈值: {validator.threshold}")

        # 创建模拟结果
        mock_result = BacktestResult(
            initial_balance=100000.0,
            final_balance=120000.0,
            total_return=20.0,
            total_trades=50,
            win_trades=30,
            loss_trades=20,
            win_rate=0.6,
            max_drawdown=15.0,
            sharpe_ratio=1.5,
            annual_return=12.0,
            trades=[],
            equity_curve=pd.Series([])
        )

        # 验证结果
        validations = validator.validate(mock_result)

        logger.info(f"\n✅ 验证结果:")
        for key, value in validations.items():
            status = "✅ 通过" if value else "❌ 失败"
            logger.info(f"   {key}: {status}")

        return all(validations.values())

    except Exception as e:
        logger.error(f"❌ 回测验证器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backtest_runner():
    """测试回测运行器"""
    logger.info("\n" + "="*60)
    logger.info("📋 测试 4: 回测运行器")
    logger.info("="*60)

    try:
        from tests.backtest.backtest_runner import BacktestRunner
        from tests.backtest.example_strategies import combined_strategy

        # 创建运行器
        runner = BacktestRunner(
            initial_balance=100000.0,
            start_dt=datetime(2020, 1, 1),
            end_dt=datetime(2020, 6, 30)
        )

        logger.info("✅ 回测运行器创建成功")
        logger.info(f"   初始资金: {runner.initial_balance:,.2f}")
        logger.info(f"   时间范围: {runner.start_dt} 到 {runner.end_dt}")

        # 运行单个策略
        logger.info("\n   运行组合策略...")
        result = runner.run_strategy(
            strategy_func=combined_strategy,
            symbol='SHFE.rb',
            period='1d',
            verbose=True
        )

        if result:
            logger.info(f"\n✅ 策略运行成功: {result.final_balance:,.2f} (收益 {result.total_return:.2f}%)")
            return True
        else:
            logger.error("❌ 策略运行失败")
            return False

    except Exception as e:
        logger.error(f"❌ 回测运行器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    logger.info("\n" + "="*80)
    logger.info("🎯 回测系统测试套件")
    logger.info("="*80)

    results = []

    # 运行所有测试
    results.append(("tqsdk 连接", test_tqsdk_connection()))
    results.append(("回测引擎", test_backtest_engine()))
    results.append(("回测验证器", test_backtest_validator()))
    results.append(("回测运行器", test_backtest_runner()))

    # 打印总结
    logger.info("\n" + "="*80)
    logger.info("📊 测试总结")
    logger.info("="*80)

    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"   {test_name}: {status}")

    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)

    logger.info(f"\n总计: {passed_tests}/{total_tests} 通过 ({passed_tests/total_tests*100:.1f}%)")

    if passed_tests == total_tests:
        logger.info("\n🎉 所有测试通过！")
        return 0
    else:
        logger.error("\n⚠️ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
