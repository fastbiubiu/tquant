"""
Tquant - 量化交易系统主入口
"""

import sys
from .main import TradingSystem


def main():
    """Main entry point for the tquant system."""
    try:
        # CLI 命令列表
        cli_commands = ['version', 'config_show', 'config_path', 'config_init', 'config_validate']

        # Check if running with a CLI command
        if len(sys.argv) > 1 and sys.argv[1] in cli_commands:
            # Remove the command from sys.argv
            sys.argv = sys.argv[1:]

            # Import typer and run the CLI
            from . import cli
            cli.app()
            return

        # Run the trading system directly
        print("Connecting to tqsdk API...")
        ts = TradingSystem(config_path=None)  # Will use default config

        # Connect API before starting
        if ts.api:
            success = ts.api.connect()
            if not success:
                print("❌ 无法连接到 tqsdk API")
                print("请配置 TQUANT_TQSDK_AUTH 环境变量或编辑 ~/.tquant/config.json")
                print("注册地址: https://account.shinnytech.com/")
                sys.exit(1)
        else:
            print("❌ 无法创建 tqsdk API 实例")
            sys.exit(1)

        # Add signal handlers
        import signal
        def signal_handler(signum, frame):
            print("\nReceived signal, stopping...")
            ts.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Run the system
        print("Starting Tquant trading system...")
        ts.start()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
