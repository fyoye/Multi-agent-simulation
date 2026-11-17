import json
import numpy as np
import pandas as pd
import pytz
from matplotlib.dates import DateFormatter
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from datetime import datetime
import networkx as nx
from typing import Dict, List, Tuple
import os

# 设置中文字体
#plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用黑体显示中文
#plt.rcParams["font.family"] = ["Noto Sans CJK SC", "Noto Serif CJK SC", "sans-serif"]
#plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 1. 配置 Noto CJK 字体文件路径（步骤1获取的实际路径）
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 替换为你的路径

# 2. 验证字体文件是否存在
if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(f"字体文件不存在，请检查路径：{FONT_PATH}")

# 3. 强制将字体添加到 matplotlib 字体管理器（关键：绕过环境隔离）
fm.fontManager.addfont(FONT_PATH)
font_prop = fm.FontProperties(fname=FONT_PATH)
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 4. 清除 matplotlib 缓存（避免旧配置干扰）
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "matplotlib")
if os.path.exists(cache_dir):
    import shutil
    shutil.rmtree(cache_dir)
    print("✅ 已清除 matplotlib 缓存")

# 5. 验证配置是否生效
available_fonts = [f.name for f in fm.fontManager.ttflist]
if font_prop.get_name() in available_fonts:
    print(f"✅ 成功加载字体：{font_prop.get_name()}")
else:
    plt.rcParams["font.family"] = font_prop.get_name()
    print(f"⚠️  字体未加入管理器，但已强制指定：{font_prop.get_name()}")

class NumpyJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


class EmotionAnalysisModule:

    def __init__(self, config):
        self.config = config
        self.output_path = config['data_path']['analysis_output']
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def analyze_emotion_timeline(self, group_stats_history: List[Dict]) -> pd.DataFrame:
        """分析情绪时间线数据，生成结构化DataFrame"""
        timeline_data = []

        for stats in group_stats_history:
            timestamp = stats.get('timestamp', 0)

            if isinstance(timestamp, str):
                try:
                    from datetime import datetime as dt
                    timestamp = dt.fromisoformat(timestamp).timestamp()
                except:
                    try:
                        timestamp = float(timestamp)
                    except:
                        timestamp = 0
            elif hasattr(timestamp, 'timestamp'):
                timestamp = timestamp.timestamp()
            else:
                timestamp = float(timestamp)

            phase = stats.get('phase', 'unknown')
            emotions = stats.get('emotion_averages', {})

            row = {
                'timestamp': timestamp,
                'datetime': stats.get('datetime', datetime.fromtimestamp(timestamp).isoformat()),
                'phase': phase,
                'anger_avg': float(emotions.get('anger', 0)),
                'anxiety_avg': float(emotions.get('anxiety', 0)),
                'disappointment_avg': float(emotions.get('disappointment', 0)),
                'trust_avg': float(emotions.get('trust', 0.5)),
                'positivity_avg': float(emotions.get('positivity', 0.5)),
                'high_anger_count': stats.get('high_emotion_counts', {}).get('anger', 0),
                'high_anxiety_count': stats.get('high_emotion_counts', {}).get('anxiety', 0),
                'low_trust_count': stats.get('high_emotion_counts', {}).get('low_trust', 0),
                'negative_ratio': stats.get('social_media_state', {}).get('negative_comment_ratio', 0)
            }
            timeline_data.append(row)

        df = pd.DataFrame(timeline_data)


        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Shanghai')
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values('timestamp')

        return df

    def plot_emotion_timeline(self, df: pd.DataFrame, save_path: str = None):
        """绘制情绪时间线图：双子图展示负面情绪演化与信任度/积极情绪变化"""
        plt.figure(figsize=(14, 8))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        ax1.plot(df['timestamp'], df['anger_avg'], label='愤怒平均', color='red', linewidth=2)
        ax1.plot(df['timestamp'], df['anxiety_avg'], label='焦虑平均', color='orange', linewidth=2)
        ax1.plot(df['timestamp'], df['disappointment_avg'], label='失望平均', color='brown', linewidth=2)
        ax1.plot(df['timestamp'], df['negative_ratio'], label='负面比例',
                 color='black', linewidth=3, linestyle='--')

        ax1.set_ylabel('情绪强度', fontsize=12)
        ax1.set_title('负面情绪演化时间线', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        self._mark_phases(ax1, df)

        ax2.plot(df['timestamp'], df['trust_avg'], label='信任度平均', color='blue', linewidth=2)
        ax2.plot(df['timestamp'], df['positivity_avg'], label='积极性平均', color='green', linewidth=2)

        ax2.set_xlabel('时间', fontsize=12)
        ax2.set_ylabel('程度', fontsize=12)
        ax2.set_title('信任度与积极性变化', fontsize=14)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        self._mark_phases(ax2, df)


        date_fmt = DateFormatter('%Y-%m-%d %H:%M', tz=pytz.timezone('Asia/Shanghai'))
        ax2.xaxis.set_major_formatter(date_fmt)
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_path, 'emotion_timeline.png'),
                        dpi=300, bbox_inches='tight')
        plt.close()

    def _mark_phases(self, ax, df):
        """在图上标记实验阶段（基线/干预/干预后），用不同颜色背景区分"""
        phases = df['phase'].unique()
        phase_colors = {'baseline': 'lightgray', 'intervention': 'lightblue',
                        'post-intervention': 'lightgreen'}

        for i, phase in enumerate(phases):
            phase_df = df[df['phase'] == phase]
            if not phase_df.empty:
                start = phase_df['timestamp'].min()
                end = phase_df['timestamp'].max()
                ax.axvspan(start, end, alpha=0.2, color=phase_colors.get(phase, 'gray'),
                           label=f'{phase}阶段' if i == 0 else "")

    def analyze_emotion_contagion(self, contagion_events: List[Dict]) -> Dict:
        contagion_stats = {
            'total_events': len(contagion_events),
            'contagion_by_emotion': defaultdict(int),
            'avg_contagion_strength': 0,
            'contagion_chains': [],
            'peak_contagion_time': None
        }

        if not contagion_events:
            return contagion_stats

        total_strength = 0
        chains_map = defaultdict(list)

        for event in contagion_events:
            strength = event.get('contagion_strength', 0)
            total_strength += strength

            pre_emotion = event.get('pre_emotion', {})
            post_emotion = event.get('post_emotion', {})

            max_change = 0
            dominant_emotion = None

            for emotion in ['anger', 'anxiety', 'disappointment']:
                change = post_emotion.get(emotion, 0) - pre_emotion.get(emotion, 0)
                if change > max_change:
                    max_change = change
                    dominant_emotion = emotion

            if dominant_emotion:
                contagion_stats['contagion_by_emotion'][dominant_emotion] += 1

                agent_id = event.get('agent_id')
                if agent_id is not None:
                    chains_map[dominant_emotion].append({
                        'agent_id': agent_id,
                        'timestamp': event.get('timestamp'),
                        'strength': strength,
                        'change': max_change
                    })

        contagion_stats['avg_contagion_strength'] = total_strength / len(contagion_events)

        for emotion, chain_events in chains_map.items():
            if len(chain_events) > 1:
                sorted_events = sorted(chain_events, key=lambda x: x['timestamp'])
                contagion_stats['contagion_chains'].append({
                    'emotion': emotion,
                    'length': len(sorted_events),
                    'total_strength': sum(e['strength'] for e in sorted_events),
                    'agents': [e['agent_id'] for e in sorted_events]
                })

        time_bins = defaultdict(int)
        for event in contagion_events:
            timestamp = event.get('timestamp', 0)
            if hasattr(timestamp, 'timestamp'):
                timestamp = timestamp.timestamp()
            else:
                timestamp = float(timestamp)

            time_bin = int(timestamp // 3600) * 3600
            time_bins[time_bin] += 1

        if time_bins:
            peak_time = max(time_bins.items(), key=lambda x: x[1])[0]
            contagion_stats['peak_contagion_time'] = datetime.fromtimestamp(peak_time).isoformat()

        return contagion_stats

    def plot_emotion_heatmap(self, group_stats_history: List[Dict], save_path: str = None):
        """绘制情绪热力图"""
        time_points = []
        emotion_data = {
            'anger': [],
            'anxiety': [],
            'disappointment': [],
            'trust': [],
            'positivity': []
        }

        for stats in group_stats_history:
            timestamp = stats['timestamp']
            if hasattr(timestamp, 'timestamp'):
                timestamp = timestamp.timestamp()
            else:
                timestamp = float(timestamp)

            time_points.append(datetime.fromtimestamp(timestamp).strftime('%H:%M'))
            emotions = stats['emotion_averages']

            for emotion in emotion_data:
                emotion_data[emotion].append(float(emotions.get(emotion, 0)))

        data_matrix = []
        emotion_labels = []
        for emotion, values in emotion_data.items():
            data_matrix.append(values)
            emotion_labels.append(emotion)

        plt.figure(figsize=(14, 6))
        sns.heatmap(data_matrix,
                    xticklabels=time_points[::2],
                    yticklabels=emotion_labels,
                    cmap='RdYlBu_r',
                    cbar_kws={'label': '情绪强度'},
                    vmin=0, vmax=1)

        plt.title('群体情绪演化热力图', fontsize=14)
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('情绪维度', fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_path, 'emotion_heatmap.png'),
                        dpi=300, bbox_inches='tight')
        plt.close()


class BehaviorAnalysisModule:
    """行为模式分析模块：识别情绪驱动的行为规律与负面行为传播路径"""

    def __init__(self, config):
        self.config = config
        self.output_path = config['data_path']['analysis_output']

    def analyze_behavior_patterns(self, behavior_patterns: Dict) -> pd.DataFrame:
        """分析行为模式， 核心逻辑：将{情绪状态_行为: 次数}的原始数据，转换为情绪-行为矩阵，用于后续热力图或堆叠条形图可视化"""
        emotion_behavior_matrix = defaultdict(lambda: defaultdict(int))

        for pattern_key, count in behavior_patterns.items():
            if '_' in pattern_key:
                emotion_state, action = pattern_key.split('_', 1)
                emotion_behavior_matrix[emotion_state][action] += count

        df = pd.DataFrame(emotion_behavior_matrix).fillna(0).astype(int)
        return df

    def plot_behavior_patterns(self, df: pd.DataFrame, save_path: str = None):
        """绘制情绪状态-行为堆叠条形图，直观展示不同情绪下的行为分布"""
        plt.figure(figsize=(12, 8))

        ax = df.T.plot(kind='bar', stacked=True, figsize=(12, 8))

        plt.title('不同情绪状态下的行为模式分布', fontsize=14)
        plt.xlabel('情绪状态', fontsize=12)
        plt.ylabel('行为次数', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='行为类型', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_path, 'behavior_patterns.png'),
                        dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_negative_behavior_spread(self, trajectories: List[Dict]) -> Dict:
        """分析负面行为的传播"""
        negative_behavior_timeline = []
        agents_turned_negative = set()

        for traj in trajectories:
            agent_id = traj['user_info']['aid']

            for step in traj['trajectory']:
                if step.get('action_detail', {}).get('decision_info', {}).get('extra_response'):
                    emotion_state = step.get('action_detail', {}).get('emotion_state', {})
                    if emotion_state and any(emotion_state.get(e, 0) > 0.6
                                             for e in ['anger', 'anxiety', 'disappointment']):
                        timestamp = step['timestamp']
                        if hasattr(timestamp, 'timestamp'):
                            timestamp = timestamp.timestamp()
                        else:
                            timestamp = float(timestamp)

                        negative_behavior_timeline.append({
                            'timestamp': timestamp,
                            'agent_id': agent_id,
                            'action': step['action']
                        })
                        agents_turned_negative.add(agent_id)

        return {
            'total_negative_behaviors': len(negative_behavior_timeline),
            'agents_affected': len(agents_turned_negative),
            'timeline': sorted(negative_behavior_timeline, key=lambda x: x['timestamp'])
        }


class NetworkAnalysisModule:
    """社交网络分析模块：基于图论分析代理互动网络结构与关键节点"""

    def __init__(self, config):
        self.config = config
        self.output_path = config['data_path']['analysis_output']

    def analyze_interaction_network(self, interaction_network: Dict[int, List[int]]) -> Tuple[Dict, nx.DiGraph]:
        """分析互动网络结构，计算网络指标并识别关键节点"""
        G = nx.DiGraph()

        for source, targets in interaction_network.items():
            for target in targets:
                if G.has_edge(source, target):
                    G[source][target]['weight'] += 1
                else:
                    G.add_edge(source, target, weight=1)

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        if num_nodes > 1:
            max_possible_edges = num_nodes * (num_nodes - 1)
            actual_density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
        else:
            actual_density = 0

        stats = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': actual_density,
            'avg_degree': np.mean([d for n, d in G.degree()]) if num_nodes > 0 else 0,
            'max_possible_edges': num_nodes * (num_nodes - 1) if num_nodes > 1 else 0,
            'actual_edges': num_edges
        }

        if num_nodes > 0:
            degree_centrality = nx.degree_centrality(G)
            top_degree_nodes = sorted(degree_centrality.items(),
                                      key=lambda x: x[1], reverse=True)[:5]

            if num_nodes > 2:
                try:
                    betweenness_centrality = nx.betweenness_centrality(G)
                    top_influence_nodes = sorted(betweenness_centrality.items(),
                                                 key=lambda x: x[1], reverse=True)[:5]
                except:
                    top_influence_nodes = []
            else:
                top_influence_nodes = []

            stats['key_spreaders'] = [node for node, _ in top_degree_nodes]
            stats['key_bridges'] = [node for node, _ in top_influence_nodes]

        return stats, G

    def plot_interaction_network(self, G: nx.Graph, agent_emotions: Dict = None,
                                 save_path: str = None):
        """绘制互动网络图"""
        plt.figure(figsize=(12, 10))

        ax = plt.gca()

        pos = nx.spring_layout(G, k=1 / np.sqrt(G.number_of_nodes()) if G.number_of_nodes() > 0 else 1,
                               iterations=50)

        if agent_emotions and G.number_of_nodes() > 0:
            node_colors = []
            for node in G.nodes():
                if node in agent_emotions:
                    emotions = agent_emotions[node]
                    negative_avg = (emotions.get('anger', 0) +
                                    emotions.get('anxiety', 0) +
                                    emotions.get('disappointment', 0)) / 3
                    node_colors.append(float(negative_avg))
                else:
                    node_colors.append(0.5)
        else:
            node_colors = [0.5] * G.number_of_nodes()

        edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]

        if len(node_colors) > 0:
            nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                   node_size=300, cmap='RdYlBu_r',
                                   vmin=0, vmax=1, ax=ax)
        else:
            nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                                   node_size=300, ax=ax)

        nx.draw_networkx_edges(G, pos, width=edge_widths,
                               alpha=0.5, edge_color='gray', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        plt.title('代理互动网络图', fontsize=14)
        plt.axis('off')

        if agent_emotions and len(node_colors) > 0 and any(c != 0.5 for c in node_colors):
            sm = plt.cm.ScalarMappable(cmap='RdYlBu_r',
                                       norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            plt.colorbar(sm, label='负面情绪强度', ax=ax)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_path, 'interaction_network.png'),
                        dpi=300, bbox_inches='tight')
        plt.close()


class InterventionAnalysisModule:
    """干预效果分析模块：量化评估干预策略对负面情绪的抑制效果"""

    def __init__(self, config):
        self.config = config
        self.output_path = config['data_path']['analysis_output']

    def analyze_intervention_effectiveness(self, summary_data: Dict) -> Dict:
        """正确分析干预效果，计算综合效果得分"""
        intervention_effect = summary_data.get('intervention_effect')

        if not intervention_effect:
            return {'status': 'no_intervention'}

        anger_reduction = intervention_effect.get('anger_reduction', 0)

        analysis = {
            'anger_reduction': anger_reduction,
            'anger_reduction_percentage': intervention_effect.get('anger_reduction_percentage', 0),
            'intervention_comments': intervention_effect.get('intervention_comments', 0),
            'filtered_comments': intervention_effect.get('filtered_comments', 0),
            'effectiveness_score': 0,
            'pre_anger_avg': intervention_effect.get('pre_anger_avg', 0),
            'post_anger_avg': intervention_effect.get('post_anger_avg', 0),
            'comparison_window': intervention_effect.get('comparison_window', 'unknown'),
            'intervention_start_anger_avg': intervention_effect.get('intervention_start_anger_avg', 0),
            'intervention_end_anger_avg': intervention_effect.get('intervention_end_anger_avg', 0),
            'intervention_phase_anger_reduction': intervention_effect.get(
                'intervention_phase_anger_reduction', 0),
            'intervention_phase_anger_reduction_percentage': intervention_effect.get(
                'intervention_phase_anger_reduction_percentage', 0),
            'intervention_phase_window_points': intervention_effect.get(
                'intervention_phase_window_points', 0),
            'intervention_phase_effectiveness_score': 0
        }

        pre_anger = analysis['pre_anger_avg']
        if pre_anger > 0:
            analysis['effectiveness_score'] = max(-1.0, min(1.0, anger_reduction / pre_anger))
        elif anger_reduction > 0:
            analysis['effectiveness_score'] = 1.0
        elif anger_reduction < 0:
            analysis['effectiveness_score'] = -1.0
        else:
            analysis['effectiveness_score'] = 0.0

        phase_start_anger = analysis['intervention_start_anger_avg']
        phase_anger_reduction = analysis['intervention_phase_anger_reduction']
        if phase_start_anger > 0:
            analysis['intervention_phase_effectiveness_score'] = max(
                -1.0, min(1.0, phase_anger_reduction / phase_start_anger))
        elif phase_anger_reduction > 0:
            analysis['intervention_phase_effectiveness_score'] = 1.0
        elif phase_anger_reduction < 0:
            analysis['intervention_phase_effectiveness_score'] = -1.0
        else:
            analysis['intervention_phase_effectiveness_score'] = 0.0

        return analysis

    def compare_phases(self, group_stats_history: List[Dict]) -> pd.DataFrame:
        """比较不同阶段的指标"""
        phase_stats = defaultdict(lambda: {
            'anger': [], 'anxiety': [], 'disappointment': [],
            'trust': [], 'negative_ratio': []
        })

        for stat in group_stats_history:
            phase = stat['phase']
            emotions = stat['emotion_averages']

            phase_stats[phase]['anger'].append(float(emotions.get('anger', 0)))
            phase_stats[phase]['anxiety'].append(float(emotions.get('anxiety', 0)))
            phase_stats[phase]['disappointment'].append(float(emotions.get('disappointment', 0)))
            phase_stats[phase]['trust'].append(float(emotions.get('trust', 0.5)))

            media_state = stat.get('social_media_state', {})
            phase_stats[phase]['negative_ratio'].append(
                float(media_state.get('negative_comment_ratio', 0)))

        comparison_data = []
        for phase, metrics in phase_stats.items():
            row = {'phase': phase}
            for metric, values in metrics.items():
                if values:
                    row[f'{metric}_mean'] = float(np.mean(values))
                    row[f'{metric}_std'] = float(np.std(values))
                else:
                    row[f'{metric}_mean'] = 0.0
                    row[f'{metric}_std'] = 0.0
            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def plot_intervention_comparison(self, comparison_df: pd.DataFrame,
                                     save_path: str = None):
        """绘制干预前后对比图"""
        metrics = ['anger', 'anxiety', 'disappointment', 'trust', 'negative_ratio']
        metric_labels = ['愤怒', '焦虑', '失望', '信任度', '负面评论比例']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i]

            phases = comparison_df['phase'].tolist()
            means = comparison_df[f'{metric}_mean'].tolist()
            stds = comparison_df[f'{metric}_std'].tolist()

            x = np.arange(len(phases))
            bars = ax.bar(x, means, yerr=stds, capsize=5)

            colors = ['lightcoral', 'lightblue', 'lightgreen']
            for j, bar in enumerate(bars):
                if j < len(colors):
                    bar.set_color(colors[j])

            ax.set_xticks(x)
            ax.set_xticklabels(phases)
            ax.set_ylabel('平均值')
            ax.set_title(f'{label}的阶段对比')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

        if len(metrics) < len(axes):
            axes[-1].set_visible(False)

        plt.suptitle('干预效果对比分析', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_path, 'intervention_comparison.png'),
                        dpi=300, bbox_inches='tight')
        plt.close()


class SimulationAnalyzer:
    """综合分析器：整合多维度分析，生成完整模拟结果报告"""

    def __init__(self, config):
        self.config = config
        self.emotion_analyzer = EmotionAnalysisModule(config)
        self.behavior_analyzer = BehaviorAnalysisModule(config)
        self.network_analyzer = NetworkAnalysisModule(config)
        self.intervention_analyzer = InterventionAnalysisModule(config)

    def analyze_simulation_results(self, simulation_data: Dict):
        """执行完整的模拟结果分析"""
        print("开始分析模拟结果...")

        print("1. 分析情绪演化...")
        emotion_df = self.emotion_analyzer.analyze_emotion_timeline(
            simulation_data.get('group_stats_history', []))

        if not emotion_df.empty:
            self.emotion_analyzer.plot_emotion_timeline(emotion_df)
            self.emotion_analyzer.plot_emotion_heatmap(
                simulation_data.get('group_stats_history', []))

        contagion_stats = self.emotion_analyzer.analyze_emotion_contagion(
            simulation_data.get('emotion_contagion_events', []))

        print("2. 分析行为模式...")
        behavior_df = self.behavior_analyzer.analyze_behavior_patterns(
            simulation_data.get('behavior_patterns', {}))
        if not behavior_df.empty:
            self.behavior_analyzer.plot_behavior_patterns(behavior_df)

        print("3. 分析互动网络...")
        network_stats, G = self.network_analyzer.analyze_interaction_network(
            simulation_data.get('interaction_network', {}))

        agent_emotions = {}
        if 'group_stats_history' in simulation_data and simulation_data['group_stats_history']:
            last_stats = simulation_data['group_stats_history'][-1]
            if 'agent_emotions' in last_stats:
                agent_emotions = last_stats['agent_emotions']

        if G.number_of_nodes() > 0:
            self.network_analyzer.plot_interaction_network(G, agent_emotions)

        print("4. 分析干预效果...")
        intervention_analysis = {}
        comparison_df = None

        if 'summary' in simulation_data:
            intervention_analysis = self.intervention_analyzer.analyze_intervention_effectiveness(
                simulation_data['summary'])
            if simulation_data.get('group_stats_history'):
                comparison_df = self.intervention_analyzer.compare_phases(
                    simulation_data['group_stats_history'])
                if not comparison_df.empty:
                    self.intervention_analyzer.plot_intervention_comparison(comparison_df)

        print("5. 生成分析报告...")

        emotion_stats_dict = {}
        if not emotion_df.empty:
            numeric_columns = ['anger_avg', 'anxiety_avg', 'disappointment_avg',
                               'trust_avg', 'positivity_avg']

            for col in numeric_columns:
                if col in emotion_df.columns:
                    emotion_stats_dict[col.replace('_avg', '')] = {
                        'mean': float(emotion_df[col].mean()),
                        'std': float(emotion_df[col].std()),
                        'min': float(emotion_df[col].min()),
                        'max': float(emotion_df[col].max())
                    }

        report = self.generate_analysis_report({
            'emotion_stats': emotion_stats_dict,
            'contagion_stats': contagion_stats,
            'behavior_patterns': behavior_df.to_dict() if not behavior_df.empty else {},
            'network_stats': network_stats,
            'intervention_analysis': intervention_analysis
        })

        report_path = os.path.join(self.config['data_path']['analysis_output'],
                                   'analysis_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)

        print(f"分析完成！结果保存在：{self.config['data_path']['analysis_output']}")

        return report

    def generate_analysis_report(self, analysis_results: Dict) -> Dict:
        """生成综合分析报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'emotion_peaks': {
                    'max_anger': float(analysis_results['emotion_stats'].get('anger', {}).get('max', 0)),
                    'max_anxiety': float(analysis_results['emotion_stats'].get('anxiety', {}).get('max', 0)),
                    'max_disappointment': float(analysis_results['emotion_stats'].get('disappointment', {}).get('max', 0)),
                    'avg_trust': float(analysis_results['emotion_stats'].get('trust', {}).get('mean', 0.5)),
                    'avg_positivity': float(analysis_results['emotion_stats'].get('positivity', {}).get('mean', 0.5))
                },
                'contagion': analysis_results['contagion_stats'],
                'network': analysis_results['network_stats'],
                'intervention': analysis_results.get('intervention_analysis', {})
            },
            'detailed_results': analysis_results,
            'recommendations': self._generate_recommendations(analysis_results)
        }

        return report


    def _generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """基于分析结果生成建议"""
        recommendations = []

        if analysis_results['emotion_stats'].get('anger', {}).get('max', 0) > 0.7:
            recommendations.append("检测到高愤怒峰值，建议增强情绪安抚干预")

        if analysis_results['contagion_stats']['avg_contagion_strength'] > 0.3:
            recommendations.append("情绪传染强度较高，建议增加理性引导内容")

        if analysis_results['network_stats']['density'] > 0.3:
            recommendations.append("互动网络密集，可考虑针对关键节点进行干预")

        intervention_score = analysis_results.get('intervention_analysis', {}).get('effectiveness_score', 0)
        if intervention_score < 0:
            recommendations.append("干预反而加剧了负面情绪，建议重新设计干预策略")
        elif intervention_score < 0.5:
            recommendations.append("干预效果有限，建议调整干预策略或增加干预强度")

        return recommendations
