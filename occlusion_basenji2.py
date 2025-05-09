import tensorflow as tf
import sys
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os
from Bio import SeqIO  # 用于读取FA文件
from deepexplain.tensorflow import DeepExplain
# 加载模型
checkpoint_NAM = '/home/qxiong/light-sensitive_regulatory_elements/basenji_model/Basenji2-3K-NAM.h5'
model_NAM = load_model(checkpoint_NAM)

# 定义DNA碱基对应的独热编码
base_to_index = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}

# 将DNA序列转换为独热编码
def sequence_to_onehot(sequence, length=3000):
    """将DNA序列转换为3000x4的独热编码，自动处理长度不足的情况"""
    onehot = np.zeros((length, 4))
    for i, base in enumerate(sequence.strip().upper()):  # 截断超长序列
        if base in base_to_index:
            onehot[i] = base_to_index[base]
    return onehot[np.newaxis, ...]  # 形状 (1, 3000, 4)

# 定义贡献度分析函数
def calculate_contributions(model, sequence_onehot):
    """对不同窗口参数运行Occlusion分析并取均值"""
    contributions = []
    window_params = [
        {'window_shape': (10,4), 'step': (1,4)},
        {'window_shape': (10,4), 'step': (1,4)}
    ]

    with DeepExplain(model=model) as de:
        xi = tf.cast(sequence_onehot, dtype=tf.float32)
        logits = model(xi)
        yi = tf.ones_like(logits)  # 假设预测目标为正向贡献
        for params in window_params:
            occlu = de.explain(
                'occlusion',
                logits * yi,
                model.inputs[0],  # 输入占位符
                xi,
                **params
            )
            # 确保输出为(1, 3000)，取第一个样本
            occlu = np.sum(occlu, axis=-1)[0]
            contributions.append(occlu)

    # 计算三个结果的均值
    return np.mean(contributions, axis=0)

# 批量处理FA文件中的序列
def process_fa_file(fa_file, output_dir):
    gene_order = []
    contribution_matrix = []
  
    # 读取FA文件
    for idx, record in enumerate(SeqIO.parse(fa_file, "fasta")):
        gene_id = record.id
        sequence = str(record.seq)
        print(len(sequence))
      
        # 转换为独热编码
        onehot = sequence_to_onehot(sequence)
      
        # 计算贡献度分数
        scores = calculate_contributions(model_NAM, onehot)
      
        # 保存结果
        gene_order.append(gene_id)
        contribution_matrix.append(scores)
        print(f"Processed gene {gene_id} ({idx+1})")
      
        # 保存贡献度分数
        np.save(os.path.join(output_dir, f'{gene_id}.npy'), scores)

    # 保存基因顺序文件
    np.savetxt(os.path.join(output_dir, 'gene_order.txt'), gene_order, fmt='%s')
  
    print(f"Done! Files saved in {output_dir}")


fa_file = ''  # 替换为实际FA文件路径
output_dir = ''  # 输出目录
process_fa_file(fa_file, output_dir)
