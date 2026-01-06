# Light-A-Video Sim2Real DDP 推理说明

本文档介绍如何使用 `inference_sim2real_ddp.py` 脚本进行批量的 Sim2Real（模拟转真实）视频生成。

## 1. 代码库分析

本项目 `Light-A-Video` 是一个基于扩散模型的视频重打光（Relighting）和生成的一致性框架。

*   **核心模型**:
    *   **AnimateDiff**: 用于保持视频的时序一致性，通过 `MotionAdapter` 模块实现。
    *   **IC-Light**: 用于控制图像/视频的光照条件 (`src/ic_light.py`)。它通过修改 UNet 的输入层（Concat Condition）来注入光照信息。
    *   **Stable Diffusion**: 基础文生图模型 (默认 `realistic-vision-v51`)。

*   **工作原理**:
    *   本项目没有使用传统的 ControlNet (如 Canny/Depth) 来控制结构，而是采用了 **Video-to-Video (Vid2Vid)** 的方式。
    *   即 SDEdit (Stochastic Differential Editing) 范式：将原视频加噪到一定程度（由 `strength` 参数控制），然后以此为起点进行去噪生成。
    *   `strength` 越小，生成的视频结构越接近原视频（Sim 数据），但风格化程度越低；`strength` 越大，风格化越强（Real 感更强），但可能丢失原视频的动作或结构。

*   **本次修改**:
    *   新增了 `inference_sim2real_ddp.py`，专门用于多卡并行推理。它封装了原有的 `lav_relight.py` 核心逻辑，并添加了数据切分和 DDP 初始化代码。

## 2. 环境准备

确保你已经安装了所需的依赖包。我们在 `requirements.txt` 中指定了 `moviepy==1.0.3` 以避免兼容性问题。

```bash
pip install -r requirements.txt
```

## 3. 数据准备

请准备两个文件夹：

1.  **视频文件夹** (`--video_folder`): 存放你的原始 CG/Sim 视频 (支持 `.mp4`, `.gif`)。
2.  **提示词文件夹** (`--prompt_folder`): 存放对应的文本提示词 `.txt` 文件。
    *   **注意**: 对应关系通过**文件名**匹配。例如视频是 `car_drift.mp4`，脚本会去查找 `car_drift.txt`。
    *   如果找不到对应的 txt 文件，将使用默认提示词 `"best quality"`。

**目录结构示例**:
```
my_dataset/
├── videos/
│   ├── video1.mp4
│   ├── video2.gif
│   └── ...
└── prompts/
    ├── video1.txt  (内容示例: "A realistic red sports car drifting on a race track, 4k, high quality")
    ├── video2.txt
    └── ...
```

## 4. 运行推理 (DDP 多卡)

使用 `torchrun` 启动脚本以利用你的 8 张 A6000 显卡。

```bash
# 示例：使用 8 卡并行推理
torchrun --nproc_per_node=8 inference_sim2real_ddp.py \
  --video_folder /path/to/your/input/videos \
  --prompt_folder /path/to/your/input/prompts \
  --output_folder ./results_sim2real \
  --strength 0.5 \
  --bg_source RIGHT
```

### 关键参数说明

| 参数 | 说明 | 默认值 | 建议调整 |
| :--- | :--- | :--- | :--- |
| `--strength` | **重绘强度 (0.0 - 1.0)**。Sim2Real 的核心参数。<br>值越大，越像真实世界，结构变化越大。<br>值越小，越保留 CG 原貌。 | `0.5` | 建议尝试 `0.4` - `0.7` |
| `--bg_source` | **光源方向**。可选 `NONE` (无特定), `LEFT`, `RIGHT`, `TOP`, `BOTTOM`。 | `RIGHT` | 根据视频内容选择 |
| `--n_prompt` | 负向提示词 (Negative Prompt)。 | `"bad quality..."` | 可根据需要添加 "cg, cartoon, 3d render" 等来增强真实感 |
| `--num_step` | 采样步数。 | `25` | `30-50` 质量可能更好，但速度变慢 |
| `--seed` | 随机种子。 | `42` | 固定种子可复现结果 |

## 5. 结果查看

推理完成后，结果视频将保存在 `--output_folder` 指定的目录中，文件名为 `sim2real_{原视频名}.mp4`。
