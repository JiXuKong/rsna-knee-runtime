# rsna-knee-runtime

RSNA Knee MRI **训练 / 推理提交**。标签提取见 [rsna-knee-labels](../rsna-knee-labels)。

## Kaggle 提交（单文件，复制到 Notebook）

1. 打开 `submit.py`，修改顶部路径常量：
   - `DATA_ROOT` — 竞赛数据
   - `WEIGHTS_PATH` — 你的 `.pt` 权重
   - `DINOV2_PATH` — DINOv2 backbone
   - `OUTPUT_PATH` — 输出 `submission.csv`

2. 将整个 `submit.py` 复制到 Notebook 单元格，或：

```python
# 若已上传 submit.py 到 Kaggle
exec(open("/kaggle/working/submit.py").read())
```

或：

```python
!python /kaggle/working/submit.py
```

**不依赖** `runtime/`、`scripts/` 等其它文件。

## 本地训练

```bash
python scripts/train_dino.py \
  --data-root ./data \
  --labels ../rsna-knee-labels/derived_labels.csv \
  --dinov2 /path/to/dinov2-small \
  --save ./outputs/model.pt
```

## 结构

```
submit.py          # Kaggle 入口：推理 + 写 submission.csv
scripts/train_dino.py
runtime/           # 训练 / 推理共用库
```
