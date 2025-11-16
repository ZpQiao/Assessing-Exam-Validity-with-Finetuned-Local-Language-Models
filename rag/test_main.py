import sys
import os
from pathlib import Path

# 确保可以导入模块
sys.path.append(str(Path(__file__).parent))

from test_framework import (
    ProbabilityTestFramework, 
    load_jsonl_data, 
    compare_models
)
from rag import ProbabilityRAG # 🎯 只导入一次

if __name__ == "__main__":
    print("="*80)
    print("概率论题库测试框架 v3.0 + RAG")
    print("✅ 支持checkpoint和续传")
    print("✅ 支持本地模型加载")
    print("✅ 支持RAG增强")
    print("="*80)
    
    # 1. 加载测试数据
    print("\n加载测试数据...")
    try:
        data = load_jsonl_data('probability_test_set.jsonl')
    except FileNotFoundError:
        print("❌ 找不到 probability_test_set.jsonl")
        sys.exit(1)
    
    # 2. 初始化RAG系统
    USE_RAG = True
    rag_system = None
    
    if USE_RAG:
        print("\n" + "="*60)
        print("初始化RAG系统...")
        print("="*60)
        
        try:
            rag_system = ProbabilityRAG(
                knowledge_base_path='probability_augmented_train_set.jsonl',
                language='english',
                embedding_model='Alibaba-NLP/gte-multilingual-base'
            )
            
            # 设置知识点权重
            try:
                rag_system.set_knowledge_weights(
                    csv_path="Qwen3-14B_trainset_kp_accuracy.csv", 
                    alpha=0.7
                )
            except FileNotFoundError:
                print("⚠️ 找不到知识点权重文件，跳过权重设置")
            
            print("✓ RAG系统初始化完成！")
            
        except Exception as e:
            print(f"✗ RAG系统初始化失败: {e}")
            import traceback
            traceback.print_exc()
            print("\n降级为基础模式（不使用RAG）")
            USE_RAG = False
            rag_system = None
    
    # 3. 定义测试配置
    QUICK_TEST = False
    test_limit = 10 if QUICK_TEST else None
    
    test_languages = ['english']  # 先只测试英文
    
    model_configs = [
        # Baseline - 不使用RAG
        {
            'name': 'Qwen/Qwen3-14B',
            'local_path': '/root/models/qwen3-14b-q4',
            'use_quantization': True,
            'quantization_bits': 4,
            'temperature': 0.0,
            'use_rag': False,
        },
        
        # RAG-enhanced - 使用RAG (k=3)
        {
            'name': 'Qwen/Qwen3-14B',
            'local_path': '/root/models/qwen3-14b-q4',
            'use_quantization': True,
            'quantization_bits': 4,
            'temperature': 0.0,
            'use_rag': True,
            'rag_system': rag_system,
            'rag_k': 3,
        },
    ]
    
    # 4. 运行测试
    print(f"\n配置:")
    print(f"  测试模式: {'快速测试 (前' + str(test_limit) + '题)' if QUICK_TEST else '完整测试'}")
    print(f"  测试语言: {', '.join(test_languages)}")
    print(f"  模型数量: {len(model_configs)}")
    print(f"  RAG状态: {'启用' if USE_RAG else '禁用'}")
    
    try:
        results = compare_models(
            data=data,
            model_configs=model_configs,
            languages=test_languages,
            output_dir='./probability_test_results',
            checkpoint_dir='./checkpoints',
            limit=test_limit,
            resume=False
        )
        
        print("\n" + "="*80)
        print("所有测试完成！")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()