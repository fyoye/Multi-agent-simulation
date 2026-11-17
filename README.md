# **负面信息攻击下社交媒体群体情绪的动态干预策略研究——基于大语言模型多智能体仿真**

本项目用于研究负面信息攻击情境下的社交媒体群体情绪演化及动态干预策略。研究设计参考了 TrendSim等大语言模型（LLM）多智能体社交媒体仿真研究的思路，本项目重点关注异质攻击、用户互动、情绪传染与干预反馈之间的关系，研究异质负面信息攻击下的群体情绪演化及动态干预。系统结合大语言模型（LLM）实现五维情绪状态更新、群体状态监测，以及辟谣、情绪安抚、理性引导三类干预策略。

## 1. 项目结构

| 文件 | 功能 |
| --- | --- |
| `run_enhanced.py`、`config_enhanced.py` | 运行入口与实验配置 |
| `agent_enhanced.py`、`attacker_enhanced.py`、`intervention_agent.py` | 普通用户、异质攻击者与干预主体 |
| `social_media_enhanced.py`、`simulator_enhanced.py` | 社交媒体交互环境、事件调度和群体状态监测 |
| `emotion_analyzer.py`、`llm.py`、`mechanism.py` | 情绪更新、LLM 调用及交互机制 |
| `recorder.py`、`analysis_module.py`、`utils_enhanced.py`、`exception.py` | 运行记录、结果分析、数据处理和辅助功能 |

## 2. 环境准备

建议使用 Python 3.11，在项目根目录安装依赖：

```bash
python -m pip install numpy scipy pandas matplotlib seaborn networkx pytz requests
```

**中文字体配置：** 当前代码会在导入 `analysis_module.py` 时检查 `FONT_PATH`。运行前须将其中的路径修改为本机存在的中文字体文件，例如 Windows 下可设为：

```python
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
```

注意：即使使用 `--no-analysis`，程序仍会导入该模块。该模块还包含清理本机 Matplotlib 字体缓存的代码，使用前可按需检查。

## 3. 模型配置

**LLM 配置：** 在 `config_enhanced.py` 中设置 `LLM_config`：

```python
'LLM_config': {
    'model_name': 'glm-4-air',
    'API_KEY': '',  # 在此配置自己的密钥
    'use_mock': False
}
```

当前代码连接智谱 API，生成参数为 `temperature=0.7`、`top_p=0.9`、`max_tokens=150`，请求超时为 30 秒，最多尝试 3 次。其他模型服务商需要适配 `llm.py` 的接口。

**模型说明：**默认模型配置更新为`glm-4-air`。

未配置 API Key 或设置 `use_mock=True` 时，程序使用模拟 LLM；真实 API 请求失败时也可能返回备用响应。

## 4. 运行示例

以下命令均使用 `medium` 攻击强度。为避免覆盖结果，两个实验使用不同的输出及检查点路径。

**无干预：**

```bash
python run_enhanced.py --degree medium --baseline full --intervention none -record_path output/no_intervention/record.json -analysis_output output/no_intervention/analysis/ -save_checkpoint checkpoint/no_intervention.pickle
```

**三类组合干预：**

```bash
python run_enhanced.py --degree medium --baseline full --intervention all -record_path output/intervention/record.json -analysis_output output/intervention/analysis/ -save_checkpoint checkpoint/intervention.pickle
```

常用选项：

| 参数 | 可选值或作用 |
| --- | --- |
| `--degree` | `mild` / `medium` / `severe`：攻击者占比 10% / 20% / 30% |
| `--baseline` | `full`：包含攻击者；`clean`：无攻击者 |
| `--intervention` | `none` / `rumor` / `emotion` / `rational` / `all` |
| `-tweet_index` | 指定话题索引，例如 `-tweet_index 0` |
| `--no-analysis` | 不运行结束后的分析流程 |
| `--checkpoint` | 尝试从配置的检查点恢复运行 |

随机种子默认是 `42`，可在 `config_enhanced.py` 中修改，多种子实验应使用不同种子和独立的输出路径。

## 5. 输出与注意事项

运行结果包括 `record.json`、`trajectories.json`、`group_stats.json`、`emotion_timeline.csv`，以及分析目录中的 `analysis_report.json` 和相关图表（按实际运行情况生成）。

相同随机种子也不保证远程 LLM 输出完全一致。





