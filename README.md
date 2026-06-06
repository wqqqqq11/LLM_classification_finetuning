# LLM Classification Finetuning
## 项目结构

```
LLM_classification_finetuning/
├── requirements.txt              # 依赖列表
├── README.md                     # 项目说明
├── main.py                       # 主入口
├── docs/                         # 文档目录
│   ├── competition_instructions.md   # 比赛规则说明
│   └── plan                      # 执行计划
├── kaggle/                       # Kaggle 相关文件
│   ├── input/                    # 数据集与预训练模型
│   │   ├── competitions/
│   │   │   └── llm-classification-finetuning/
│   │   │       ├── train.csv                 # 训练数据 (57,477 行)
│   │   │       ├── test.csv                  # 测试数据
│   │   │       └── sample_submission.csv     # 提交样例
│   │   └── models/
│   │       └── emiz6413/
│   │           └── gemma-2/
│   │               └── transformers/
│   │                   └── gemma-2-9b-it-4bit/
│   │                       └── 1/
│   │                           └── gemma-2-9b-it-4bit/
│   │                               ├── config.json
│   │                               ├── tokenizer.json
│   │                               ├── tokenizer_config.json
│   │                               ├── tokenizer.model
│   │                               ├── special_tokens_map.json
│   │                               ├── model.safetensors.index.json
│   │                               ├── model-00001-of-00002.safetensors
│   │                               └── model-00002-of-00002.safetensors
│   └── working/                  # Kaggle Notebook 工作目录
│       └── src/                  # 核心源码
│           ├── config.py         # 全局配置管理
│           ├── data/             # 数据处理模块
│           │   ├── dataset.py        # Dataset 封装
│           │   ├── preprocessor.py   # 文本格式化与截断
│           │   └── augmentation.py   # 数据增强
│           ├── models/           # 模型定义
│           │   ├── base_model.py     # 基类接口
│           │   └── gemma_classifier.py   # Gemma + LoRA 实现
│           ├── training/         # 训练模块
│           │   ├── trainer.py        # 训练循环
│           │   └── evaluation.py     # 评估指标 (Log Loss)
│           ├── inference/        # 推理模块
│           │   └── predictor.py      # 预测与生成提交文件
│           └── utils/            # 工具函数
│               └── logging.py  # 日志管理
│
├── outputs/                      # 输出结果
│   ├── data_analysis/            # 数据分析报告
│   └── logs/                     # 训练日志
│
└── tools/                        # 工具脚本
    └── analyze_data.py           # 数据分析脚本
```

# 训练模式
python main.py --mode train

# 预测模式（需要指定模型路径）
python main.py --mode predict --model_path outputs/training/final