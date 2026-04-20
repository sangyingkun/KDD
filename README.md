# KDD Data Agent Baseline

KDD Cup 2026 数据智能体基线代码。

## 版本

**1.0.1**

## 功能特性

- ReAct Agent 框架
- 语义层（Semantic Layer）
- 中文日志系统
- SQLite/Python 工具支持

## 快速开始

```bash
# 安装依赖
pip install -e .

# 运行基线
python -m data_agent_baseline --config configs/react_baseline.yaml
```

## 目录结构

```
src/
├── agents/      # Agent 核心逻辑
├── tools/       # 工具注册与实现
├── semantic/    # 语义层
├── benchmark/   # 评测相关
└── run/         # 运行器
```

## 协作开发

```bash
git clone https://github.com/sangyingkun/KDD.git
git pull
git add .
git commit -m "描述"
git push
```
