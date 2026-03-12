# Unlearning Task Vector Heat Map

基于 `pareto_comparison.json` 中的 retain90、unlearned 模型与 base 模型，计算 task vector 并绘制余弦相似度热力图。

## Task Vector 定义

τ = θ_model − θ_base

- **base model**: `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`
- **retain90**: `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90`（参考基准）
- **unlearned models**: GradDiff pareto 变体（lr2e-05_α1_ep5, lr2e-05_α5_ep5, ...）

## 用法

```bash
# 默认使用 ../unlearning-wikitext-filter/output_pareto/pareto_comparison.json
python task_vector_heatmap.py

# 指定 pareto JSON 路径
python task_vector_heatmap.py --pareto_json /path/to/pareto_comparison.json

# 指定输出目录
python task_vector_heatmap.py --output_dir ./output
```

## 输出

- `output/task_vector_heatmap.png` — 余弦相似度热力图
- `output/task_vector_similarity.json` — 相似度矩阵及统计

## 参考

- `unlearning-low-rank/explore_open_unlearning_models.ipynb` — Task Vector 余弦相似度分析
- [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) Figure 5
