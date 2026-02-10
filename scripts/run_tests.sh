#!/bin/bash

# tquant 测试运行脚本

set -e

echo "🧪 开始运行测试..."

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  未激活虚拟环境，正在激活..."
    source venv/bin/activate
fi

# 定义颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 运行单元测试
echo -e "${YELLOW}📋 运行单元测试...${NC}"
python -m pytest tests/unit/ -v --tb=short --cov=src --cov-report=html:htmlcov/unit

# 运行集成测试
echo -e "${YELLOW}📋 运行集成测试...${NC}"
python -m pytest tests/integration/ -v --tb=short

# 运行性能测试
echo -e "${YELLOW}📋 运行性能测试...${NC}"
python -m pytest tests/performance/ -v --tb=short

# 运行回测测试
echo -e "${YELLOW}📋 运行回测测试...${NC}"
python -m pytest tests/backtest/ -v --tb=short

# 生成覆盖率报告
echo -e "${YELLOW}📊 生成覆盖率报告...${NC}"
python -m pytest tests/ --cov=src --cov-report=html:htmlcov --cov-report=term-missing

echo ""
echo -e "${GREEN}✅ 所有测试完成！${NC}"
echo ""
echo "覆盖率报告已生成到 htmlcov/index.html"
