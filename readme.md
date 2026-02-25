# nnssl-seg 

## 1. 关键文件与路径清单 (File Locations)
| 项目 | 绝对路径 |
| :--- | :--- |
| **项目根目录** | `/gpfs/share/home/2401111663/syy/nnssl-openneuro` |
| **Python 环境** | `/gpfs/share/home/2401111663/anaconda3/envs/syy1/bin/python` |
| **预训练权重** | `/gpfs/share/home/2401111663/syy/nnssl-openneuro/checkpoint_final_mae.pth` |
| **JSON 生成脚本** | `src/nnssl/utilities/prepare_ped_v3_json_nnssl.py` |
| **数据 Datalist** | `/gpfs/share/home/2401111663/syy/braTS_5folds/` |
| **Slurm 脚本** | `sbatch/nnssl_ped_v3_f2to5.sh` |
| **训练日志** | `sbatch/output/ped/` |
| **模型保存** | `training_runs/v3_mae/` |

## 权重文件说明

- **外部预训练权重**: 本项目默认使用 `/gpfs/share/home/2401111663/syy/nnssl-openneuro/checkpoint_final_mae.pth` 作为 `ResEncL` 主干网络的预训练权重。
- **本地训练输出（不纳入仓库）**: 训练产生的模型文件（如 `best_model.pth`）保存在 `training_runs/` 目录下，这些文件已被列入 `.gitignore`，不会提交到代码仓库。

## 2. 环境配置
- **Python**: 3.10+ (使用 conda 环境 `syy1`)
- **关键依赖**: `torch`, `monai`, `einops`, `cc3d`
- **环境变量**: 运行前需确保 `PYTHONPATH` 包含 `src` 目录。

## 3. 数据列表生成 (JSON Generation)
在微调训练前，需将分折原始 JSON 转换为项目专用格式。V3 实验集成了自动切分和合成数据注入功能。

### A. V3 数据处理脚本 (prepare_ped_v3_json_nnssl.py)
该脚本专门用于 V3 实验的数据准备，将 BraTS 字典格式转换为 nnSSL 输入格式。
- **脚本位置**: `src/nnssl/utilities/prepare_ped_v3_json_nnssl.py`
- **主要逻辑与模态筛选**:
  1. **T1c 模态提取**: 脚本专门寻找 `T1-weighted Contrast Enhanced` (或缩写 `CE`) 路径。V3 实验采用单通道输入。
  2. **训练/验证二次切分**: 
     - 将原始 `train.json` 随机打乱后，划分出固定的 **10 例真实数据** 作为验证集 (`validation`)。
     - 剩余数据作为训练集 (`training`)。
  3. **测试集确定**: 原始的 `val.json` 固定作为测试集 (`test`)。
  4. **合成数据注入 (Mixed 类型专用)**: 脚本会匹配后缀为 `_generated.nii.gz` 的文件，并将其追加到 **Mixed** 版本的训练集中。

### B. 生成json命令
sbatch 脚本中已集成了自动生成逻辑，核心调用命令如下：
```bash
python src/nnssl/utilities/prepare_ped_v3_json_nnssl.py --json-in [IN_JSON] --val-json [VAL_JSON] --output-json [OUT_JSON]
```

## 4. 实验流程
1. **预处理 JSON**: 使用提供的脚本生成 `jsons/` 下的专用 datalist。
2. **配置 sbatch**: 修改 `sbatch/nnssl_ped_v3_f2to5.sh` 中的折数或路径。
3. **提交任务**: 
   ```bash
   sbatch sbatch/nnssl_ped_v3_f2to5.sh
   ```
4. **结果**: 通过 `sbatch/output/ped/` 下的日志查看 Dice 和相关指标。

## 5. 数据集结构说明
数据列表存储在 `/gpfs/share/home/2401111663/syy/braTS_5folds/` 下：
- **Pediatric (PED)**: 5 折交叉验证。

## 6. 训练参数说明
在 `src/nnssl/evaluation/segmentation3d_ped_v3.py` 中，关键参数包括：
- `--datalist-path`: 指定生成的 nnSSL JSON 路径。
- `--pretrained-weights`: 预训练模型路径。
- `--epochs`: 默认 150 轮。
- `--batch-size`: 默认 2。
- `--image-size`: 96 x 96 x 96。
- `--learning-rate`: 0.0003。



