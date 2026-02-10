# tquant - 量化交易系统

一个基于技术指标和 LLM 的混合架构量化交易系统，专为**中国期货市场**设计，支持自动交易和风险控制。

## 🎯 核心特性

- **混合架构**：结合传统技术指标与 LLM 增强决策
- **技术分析**：8 种技术指标实时计算
- **LLM 增强**：GPT-4o/GPT-4o-mini 智能信号优化
- **多 Agent 系统**：市场分析、交易执行、信号增强、辩论机制
- **风险控制**：止损、止盈、仓位管理、每日亏损限制
- **实时监控**：交易监控、通知系统（Telegram/微信/邮件）
- **异步处理**：高性能异步架构，低延迟决策
- **成本优化**：智能缓存、API 请求优化

## 📋 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    系统架构图                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐                                           │
│  │  市场分析     │  (8 种技术指标)                            │
│  │  生成基础信号 │                                           │
│  └──────┬───────┘                                           │
│         ↓                                                   │
│  ┌──────────────┐                                           │
│  │  LLM 增强     │  (GPT-4o/GPT-4o-mini)                     │
│  │  信号优化     │                                           │
│  └──────┬───────┘                                           │
│         ↓                                                   │
│  ┌──────────────┐                                           │
│  │  拉锯辩论     │  (多角度分析，风险控制)                     │
│  │  生成决策     │                                           │
│  └──────┬───────┘                                           │
│         ↓                                                   │
│  ┌──────────────┐                                           │
│  │  交易执行     │  (tqsdk API)                              │
│  │  风险管理     │                                           │
│  └──────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 环境设置

```bash
# 克隆项目
git clone <repository-url>
cd tquant

# 设置虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 安装项目及其依赖
pip install -e .
```

### 2. 配置系统

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env` 文件填入配置：

```bash
# OpenAI 配置
OPENAI_API_KEY=your_api_key
OPENAI_ORGANIZATION_ID=your_org_id

# tqsdk 账户配置
TQSDK_ACCOUNT=your_account
TQSDK_PASSWORD=your_password
TQSDK_URL=tqs://quant.zts.com

# 交易参数
TQUANT_MAX_TRADES_PER_DAY=10
TQUANT_MAX_POSITION_SIZE=100
TQUANT_LEVERAGE=10

# 风险控制
TQUANT_STOP_LOSS_PCT=0.03
TQUANT_TAKE_PROFIT_PCT=0.06
TQUANT_DAILY_LOSS_LIMIT=0.1

# 通知配置
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
WECHAT_WEBHOOK_URL=your_url
EMAIL_SMTP_HOST=smtp.example.com
EMAIL_SMTP_PORT=587
EMAIL_FROM_ADDR=from@example.com
EMAIL_FROM_PASSWORD=your_password
```

### 3. 运行系统

```bash
# 启动交易系统（交互模式）
python -m tquant

# 使用 CLI 命令
tquant version                    # 显示版本信息
tquant config_show                # 显示配置
tquant config_path                # 显示配置文件位置
tquant config_init                # 初始化配置
tquant config_validate            # 验证配置

# 或直接运行 CLI 模块
python -m tquant.cli version
python -m tquant.cli config_validate
```

## 📁 项目结构

```
tquant/
├── src/
│   ├── agents/                  # AI Agent 模块
│   │   ├── market_analyst.py    # 市场分析 Agent
│   │   ├── trader.py            # 交易执行 Agent
│   │   ├── signal_enhancer.py   # LLM 信号增强
│   │   └── debater.py           # 拉锯辩论机制
│   │
│   ├── utils/                   # 工具模块
│   │   ├── indicators.py        # 技术指标计算
│   │   ├── signals.py           # 交易信号定义
│   │   ├── tqsdk_interface.py   # tqsdk API 封装
│   │   ├── async_analyzer.py    # 异步分析器
│   │   ├── batch_processor.py   # 批量处理
│   │   ├── cache_manager.py     # 缓存管理
│   │   ├── cost_optimizer.py    # 成本优化
│   │   ├── prompt_optimizer.py  # Prompt 优化
│   │   ├── notification_manager.py  # 通知管理
│   │   ├── result_recorder.py   # 结果记录
│   │   └── trading_monitor.py   # 交易监控
│   │
│   ├── workflows/               # 工作流编排
│   │   └── workflow.py          # LangGraph 工作流
│   │
│   ├── config/                  # 配置管理
│   │   ├── schema.py            # 配置模式定义
│   │   └── loader.py            # 配置加载器
│   │
│   ├── main.py                  # 主入口点
│   └── cli.py                   # CLI 接口
│
├── tests/                       # 测试文件
│   ├── unit/                    # 单元测试 (12 个测试文件)
│   ├── integration/             # 集成测试
│   ├── performance/             # 性能测试
│   ├── backtest/                # 回测测试
│   └── run_tests.py             # 测试运行器
│
├── scripts/                     # 脚本
│   ├── run_tests.sh             # 运行测试
│   └── cleanup.sh               # 清理脚本
│
├── docs/                        # 文档
│   ├── API_DOCUMENTATION.md     # API 文档
│   └── USER_GUIDE.md            # 用户指南
│
├── data/                        # 数据文件
├── logs/                        # 日志文件
│
├── requirements.txt             # Python 依赖
├── pyproject.toml               # 项目配置
├── .env.example                 # 环境变量模板
├── ROADMAP.md                   # 项目路线图
└── COMPLETION_REPORT.md         # 完成报告
```

## 🧩 核心模块

### 1. Agent 系统

#### MarketAnalyst Agent
- 分析多个交易品种
- 计算 8 种技术指标
- 生成带置信度的交易信号
- 提供详细分析理由

**支持的技术指标：**
- MA (移动平均线) - 5, 10, 20, 60 周期
- MACD - DIF, DEA, MACD 柱状图
- RSI - 相对强弱指数
- KDJ - 随机指标
- Bollinger Bands - 布林带
- Williams %R - 威廉指标
- CCI - 商品通道指标
- DMI - 方向运动指数

#### Trader Agent
- 通过 tqsdk 执行买卖订单
- 仓位管理（多头/空头）
- 风险管理（止损、止盈）
- 投资组合跟踪
- 交易历史记录

#### SignalEnhancer Agent
- 使用 LLM 增强交易信号
- 信号验证和权重优化
- Prompt 优化提升 LLM 响应质量

#### Debater Agent
- 多空拉锯辩论机制
- 多角度市场分析
- 辩论后决策

### 2. 技术指标模块

**indicators.py** 提供完整的指标计算库：
- MA：移动平均线
- MACD：平滑异同移动平均线
- RSI：相对强弱指数
- KDJ：随机指标
- Bollinger Bands：布林带
- Williams %R：威廉指标
- CCI：商品通道指数
- DMI：方向运动指数

### 3. 工作流系统

**workflow.py** 基于 LangGraph 的状态工作流：
```
analyze_market → evaluate_signals → execute_trades → update_portfolio → monitor_risk
```

### 4. 工具模块

- **async_analyzer.py**：异步市场分析
- **batch_processor.py**：批量数据处理
- **cache_manager.py**：API 响应缓存
- **cost_optimizer.py**：API 成本优化
- **prompt_optimizer.py**：LLM Prompt 优化
- **notification_manager.py**：多渠道通知（Telegram/微信/邮件）
- **result_recorder.py**：交易结果记录
- **trading_monitor.py**：实时交易监控
- **signal_validator.py**：信号验证

## 📊 测试

### 运行所有测试

```bash
# 运行完整测试套件
bash scripts/run_tests.sh

# 或使用 pytest
python -m pytest tests/ -v --tb=short --cov=tquant --cov-report=html

# 运行特定类型测试
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/performance/ -v
python -m pytest tests/backtest/ -v
```

### 测试覆盖

- **单元测试**：12 个测试文件，覆盖核心模块
- **集成测试**：系统级测试
- **性能测试**：性能基准测试
- **回测测试**：历史数据回测

## 🛠️ 配置说明

### 配置文件优先级

1. 环境变量（TQUANT_* 前缀）
2. JSON 配置文件
3. 默认值

### 主要配置类别

| 类别 | 配置项 | 说明 |
|------|--------|------|
| **OpenAI** | OPENAI_API_KEY | API 密钥 |
| **账户** | TQSDK_ACCOUNT, TQSDK_PASSWORD | tqsdk 账户 |
| **交易** | TQUANT_MAX_TRADES_PER_DAY, TQUANT_MAX_POSITION_SIZE | 交易限制 |
| **风险** | TQUANT_STOP_LOSS_PCT, TQUANT_TAKE_PROFIT_PCT | 止损止盈 |
| **通知** | TELEGRAM_BOT_TOKEN, WECHAT_WEBHOOK_URL | 通知渠道 |
| **数据库** | DATABASE_URL | 数据库连接 |
| **缓存** | CACHE_ENABLED, CACHE_TTL | 缓存设置 |

### 配置验证

```bash
tquant config_validate
```

## 📦 依赖安装

### 方式 1：使用 pyproject.toml（推荐）

```bash
# 仅安装核心依赖（不含开发依赖）
pip install -e .

# 安装开发依赖
pip install -e ".[dev]"
```

### 方式 2：使用 pip-tools（可选）

```bash
# 生成 requirements.txt
pip-compile pyproject.toml -o requirements.txt

# 安装依赖
pip install -r requirements.txt
```

### 核心依赖（来自 pyproject.toml）

- **langchain** >= 0.1.0 - LLM 框架
- **langgraph** >= 0.0.1 - 图形化工作流
- **openai** >= 1.0.0 - OpenAI API
- **tqsdk** >= 2.3.0 - 交易 API
- **pandas** >= 2.0.0 - 数据处理
- **numpy** >= 1.24.0 - 数值计算
- **apscheduler** >= 3.10.0 - 任务调度
- **aiohttp** >= 3.9.0 - 异步 HTTP
- **pydantic** >= 2.0.0 - 配置模式
- **typer** >= 0.9.0 - CLI 框架
- **prometheus-client** >= 0.18.0 - 监控指标
- **loguru** >= 0.7.0 - 高级日志
- **rich** >= 13.0.0 - 终端输出格式化
- **python-dotenv** >= 1.0.0 - 环境变量管理
- **python-json-logger** >= 2.0.0 - JSON 日志

### 开发依赖

- pytest, pytest-asyncio, pytest-cov, pytest-mock - 测试框架
- ruff - 代码规范和格式化
- mypy - 类型检查

完整的依赖配置见 [pyproject.toml](pyproject.toml)

## 🎯 性能指标

### 系统性能目标

- **技术分析**：< 1 秒
- **LLM 分析**：5-10 秒（异步）
- **辩论决策**：15-20 秒（并行）
- **总延迟**：~30 秒

### 成本目标

- **深度分析（30%）**：$30/月（GPT-4o）
- **快速分析（70%）**：$50/月（GPT-4o-mini）
- **总成本**：~$80/月

## 📝 开发路线图

### 已完成 ✅

- Phase 1-2: MVP（市场分析 + 交易执行）
- Phase 3: LLM 信号增强
- Phase 4-5: 拉锯辩论机制
- Phase 6: 风险管理
- Phase 7: 交易执行（完成）
- Phase 8: 系统集成
- Phase 9: 测试和验证（100% 测试通过）
- Phase 10: 部署

### 详细路线图

查看 [ROADMAP.md](ROADMAP.md) 了解完整的 12 周开发计划。

## 🚀 部署

### 本地部署

```bash
# 构建
python -m build

# 安装
pip install dist/tquant-*.whl

# 运行
tquant --help
```

### 生产部署

```bash
# 运行测试
bash scripts/run_tests.sh

# 启动服务
python tquant/main.py
```

## 📚 文档

- [API 文档](docs/API_DOCUMENTATION.md)
- [用户指南](docs/USER_GUIDE.md)
- [项目路线图](ROADMAP.md)
- [完成报告](COMPLETION_REPORT.md)

## 🔒 安全与风险

⚠️ **重要提醒**：

- 本系统仅供学习和研究使用
- 期货交易存在高风险，可能导致资金损失
- 请勿用于真实资金交易
- 建议使用模拟账户进行测试
- 实施交易前请进行充分回测

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 📞 联系方式

如有问题或建议，请：
- 提交 Issue
- 发送邮件
- 加入我们的社区

---

**注意**：本项目仍在积极开发中。最后更新：2026-02-10
