#!/bin/bash

# tquant 清理脚本

set -e

echo "🧹 开始清理项目..."

# 定义颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 清理 Python 缓存
echo -e "${YELLOW}🧹 清理 Python 缓存...${NC}"
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.egg-info" -delete 2>/dev/null || true

# 清理测试缓存
echo -e "${YELLOW}🧹 清理测试缓存...${NC}"
rm -rf .pytest_cache 2>/dev/null || true
rm -rf .tox 2>/dev/null || true
rm -rf htmlcov 2>/dev/null || true
rm -rf .coverage 2>/dev/null || true

# 清理构建文件
echo -e "${YELLOW}🧹 清理构建文件...${NC}"
rm -rf build 2>/dev/null || true
rm -rf dist 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true

# 清理 IDE 文件
echo -e "${YELLOW}🧹 清理 IDE 文件...${NC}"
rm -rf .vscode 2>/dev/null || true
rm -rf .idea 2>/dev/null || true
rm -rf *.swp 2>/dev/null || true
rm -rf *.swo 2>/dev/null || true

# 清理日志文件（可选）
echo -e "${YELLOW}🧹 清理旧日志文件...${NC}"
find logs -type f -name "*.json" -mtime +7 -delete 2>/dev/null || true

# 清理临时文件
echo -e "${YELLOW}🧹 清理临时文件...${NC}"
rm -rf tmp 2>/dev/null || true
rm -rf temp 2>/dev/null || true

echo ""
echo -e "${GREEN}✅ 清理完成！${NC}"
echo ""
echo "清理项目:"
echo "  - Python 缓存文件"
echo "  - 测试缓存"
echo "  - 构建文件"
echo "  - IDE 配置"
echo "  - 临时文件"
