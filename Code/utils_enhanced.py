import json
import csv
import datetime
import pytz
from datetime import datetime
import heapq
import pickle
import random
import time
import os
from copy import deepcopy
from typing import Dict, List, Tuple, Any
import numpy as np
from collections import defaultdict

DATA_AND_BASELINE_CODE = {
    'mild': {
        'clean': (400, 0, 0, 0, 0),
        'full': (360, 10, 10, 10, 10)
    },
    'medium': {
        'clean': (400, 0, 0, 0, 0),
        'full': (320, 20, 20, 20, 20)
    },
    'severe': {
        'clean': (400, 0, 0, 0, 0),
        'full': (280, 30, 30, 30, 30)
    },
    'debug': {
        'clean': (400, 0, 0, 0, 0),
        'full': (320, 20, 20, 20, 20)
    }
}


AGENT_CLASS = {
    'normal': 'Agent',
    'S': 'SocialAttacker',
    'T': 'TrollingAttacker',
    'F': 'FactAttacker',
    'E': 'EmotionalAttacker',
    'intervention': 'InterventionAgent'
}


def update_config(cfg, extra_args):
    """更新配置，支持嵌套配置"""
    if len(extra_args) % 2 != 0:
        raise ValueError("Command argparse number is not correct!")
    
    for i in range(0, len(extra_args), 2):
        name, par = extra_args[i], extra_args[i + 1]

        if name == '-port':
            cfg['LLM_config']['port'] = int(par)
        elif name == '-tweet_index':
            cfg['media_config']['tweet_index'] = int(par)
        elif name == '-degree':
            cfg['simulation_config']['degree'] = par
        elif name == '-baseline':
            cfg['simulation_config']['baseline'] = par
        elif name == '-record_path':
            cfg['data_path']['record'] = par
        elif name == '-load_checkpoint':
            cfg['data_path']['load_checkpoint'] = par
        elif name == '-save_checkpoint':
            cfg['data_path']['save_checkpoint'] = par
        elif name == '-defense_time':
            cfg['intervention']['comment_poisoning_detect']['split_time'] = float(par)

        elif name == '-intervention_type':
            if par == 'rumor':
                cfg['intervention']['rumor_refutation']['enabled'] = True
                cfg['intervention']['emotion_soothing']['enabled'] = False
                cfg['intervention']['rational_guidance']['enabled'] = False
            elif par == 'emotion':
                cfg['intervention']['rumor_refutation']['enabled'] = False
                cfg['intervention']['emotion_soothing']['enabled'] = True
                cfg['intervention']['rational_guidance']['enabled'] = False
            elif par == 'rational':
                cfg['intervention']['rumor_refutation']['enabled'] = False
                cfg['intervention']['emotion_soothing']['enabled'] = False
                cfg['intervention']['rational_guidance']['enabled'] = True
            elif par == 'all':
                cfg['intervention']['rumor_refutation']['enabled'] = True
                cfg['intervention']['emotion_soothing']['enabled'] = True
                cfg['intervention']['rational_guidance']['enabled'] = True
            elif par == 'none':
                cfg['intervention']['rumor_refutation']['enabled'] = False
                cfg['intervention']['emotion_soothing']['enabled'] = False
                cfg['intervention']['rational_guidance']['enabled'] = False
        
        elif name == '-analysis_output':
            cfg['data_path']['analysis_output'] = par
        
        elif name == '-emotion_decay':
            cfg['agent_config']['emotion_config']['decay_rate'] = float(par)
        
        elif name == '-contagion_strength':
            cfg['agent_config']['emotion_config']['contagion_strength'] = float(par)

        elif name.startswith('-') and '.' in name:
            path = name[1:].split('.')
            current = cfg
            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            try:
                if par.lower() == 'true':
                    value = True
                elif par.lower() == 'false':
                    value = False
                elif '.' in par:
                    value = float(par)
                else:
                    value = int(par)
            except ValueError:
                value = par
            
            current[path[-1]] = value
    
    return cfg


def load_user(config):
    """加载用户，包括新的攻击者类型"""
    user_path = config['data_path']['normal_user']
    degree = config['simulation_config']['degree']
    baseline = config['simulation_config']['baseline']
    
    code = DATA_AND_BASELINE_CODE[degree][baseline]
    
    normal_user_list = load_normal_user(user_path, code)
    attacker_list = load_attacker_enhanced(code)
    
    all_user = attacker_list + normal_user_list

    random.shuffle(all_user)
    
    return all_user


def load_normal_user(path, num_tuple):
    """加载普通用户"""
    user_list = []

    if not os.path.exists(path):
        print(f"警告：用户文件 {path} 不存在，生成模拟用户")
        for i in range(num_tuple[0]):
            user_list.append({
                'type': 'normal',
                'name': f'用户{i+1}',
                'profile': {
                    'uid': str(1000000 + i),
                    'followed_num': str(random.randint(10, 10000)),
                    'following_num': str(random.randint(10, 5000)),
                    'description': generate_user_description()
                }
            })
    else:
        with open(path, 'r', encoding='UTF-8') as f:
            reader = csv.reader(f)
            for index, line in enumerate(reader):
                if index >= num_tuple[0]:
                    break

                if len(line) >= 6:
                    user_list.append({
                        'type': 'normal',
                        'name': line[2] if len(line) > 2 else f'用户{index+1}',
                        'profile': {
                            'uid': line[1] if len(line) > 1 else str(1000000 + index),
                            'followed_num': line[3] if len(line) > 3 else str(random.randint(10, 10000)),
                            'following_num': line[4] if len(line) > 4 else str(random.randint(10, 5000)),
                            'description': line[5] if len(line) > 5 else generate_user_description()
                        }
                    })
                else:
                    user_list.append({
                        'type': 'normal',
                        'name': f'用户{index+1}',
                        'profile': {
                            'uid': str(1000000 + index),
                            'followed_num': str(random.randint(10, 10000)),
                            'following_num': str(random.randint(10, 5000)),
                            'description': generate_user_description()
                        }
                    })
    
    return user_list


def generate_user_description():
    """生成随机用户描述"""
    personalities = [
        "乐观开朗，喜欢分享生活",
        "理性思考，关注社会时事",
        "温和友善，乐于助人",
        "独立思考，有自己的见解",
        "热爱生活，积极向上",
        "关注科技，理性分析",
        "文艺青年，感性细腻",
        "务实稳重，脚踏实地"
    ]
    return random.choice(personalities)


def load_attacker_enhanced(num_tuple):
    """加载增强的攻击者，包括新类型"""
    attacker_list = []
    uid_offset = num_tuple[0]

    for i in range(num_tuple[1]):
        attacker_list.append({
            'type': 'S',
            'name': f'社会批判者{i+1}',
            'profile': {
                'uid': str(uid_offset + i),
                'followed_num': str(random.randint(100, 500)),
                'following_num': str(random.randint(50, 200)),
                'description': '对社会现状极度不满，认为世界充满不公，底层人民永远被压迫。'
            }
        })
    
    uid_offset += num_tuple[1]

    for i in range(num_tuple[2]):
        attacker_list.append({
            'type': 'T',
            'name': f'争议制造者{i+1}',
            'profile': {
                'uid': str(uid_offset + i),
                'followed_num': str(random.randint(200, 800)),
                'following_num': str(random.randint(100, 400)),
                'description': '喜欢挑起争端，用尖锐言辞激怒他人，制造对立和冲突。'
            }
        })
    
    uid_offset += num_tuple[2]

    for i in range(num_tuple[3]):
        attacker_list.append({
            'type': 'F',
            'name': f'阴谋论者{i+1}',
            'profile': {
                'uid': str(uid_offset + i),
                'followed_num': str(random.randint(50, 300)),
                'following_num': str(random.randint(100, 500)),
                'description': '相信各种阴谋论，认为官方总在隐瞒真相，喜欢传播另类"事实"。'
            }
        })
    
    uid_offset += num_tuple[3]

    if len(num_tuple) > 4:
        for i in range(num_tuple[4]):
            attacker_list.append({
                'type': 'E',
                'name': f'情绪爆发者{i+1}',
                'profile': {
                    'uid': str(uid_offset + i),
                    'followed_num': str(random.randint(10, 100)),
                    'following_num': str(random.randint(20, 150)),
                    'description': '情绪极不稳定，容易愤怒、焦虑或绝望，经常失控地宣泄负面情绪。'
                }
            })
    
    return attacker_list


def load_tweet(config):
    """加载推文数据"""
    tweet_list = []
    tweet_path = config['data_path']['tweets']

    if not os.path.exists(tweet_path):
        print(f"警告：推文文件 {tweet_path} 不存在，使用默认推文")
        tweet_list = get_default_tweets()
    else:
        with open(tweet_path, 'r', encoding='UTF-8') as f:
            reader = csv.reader(f)
            for line in reader:
                if len(line) >= 6:
                    tweet_list.append({
                        'title': line[1],
                        'brief_intro': line[2],
                        'content': line[3],
                        'source': line[4],
                          'publish_time': line[5]
                    })

    if not tweet_list:
        tweet_list = get_default_tweets()

    tweet_index = config['media_config']['tweet_index']
    if tweet_index >= len(tweet_list):
        tweet_index = 0
        print(f"警告：推文索引超出范围，使用第一条推文")
    
    return tweet_list[tweet_index]


def get_default_tweets():
    """获取默认推文列表"""
    return [
        {
            'title': '某地房价再创新高引发热议',
            'brief_intro': '近日，某一线城市房价突破历史新高，引发社会各界广泛关注和讨论。',
            'content': '据最新数据显示，某市房价已连续6个月上涨，平均涨幅达15%。专家表示，这反映了供需关系的失衡。年轻人购房压力持续增大，有网友表示"工作一辈子也买不起房"。相关部门表示将出台调控措施。',
            'source': '财经新闻网',
            'publish_time': '2022-09-09 10:00:00'
        },
        {
            'title': '教育资源分配不均问题引关注',
            'brief_intro': '多地家长反映优质教育资源过度集中，普通学校与重点学校差距明显。',
            'content': '调查显示，重点学校的师资力量、教学设备明显优于普通学校。有家长为了孩子上好学校，不惜花重金购买学区房。教育专家呼吁加大教育投入，促进教育公平。但也有声音认为，完全的平均主义并不现实。',
            'source': '教育时报',
            'publish_time': '2022-09-09 09:00:00'
        }
    ]



class EmotionVector:
    """情绪向量辅助类"""
    
    def __init__(self, anger=0.1, anxiety=0.1, disappointment=0.1, 
                 trust=0.8, positivity=0.5):
        self.anger = anger
        self.anxiety = anxiety
        self.disappointment = disappointment
        self.trust = trust
        self.positivity = positivity
    
    def to_dict(self):
        return {
            'anger': self.anger,
            'anxiety': self.anxiety,
            'disappointment': self.disappointment,
            'trust': self.trust,
            'positivity': self.positivity
        }
    
    def from_dict(self, d):
        self.anger = d.get('anger', 0.1)
        self.anxiety = d.get('anxiety', 0.1)
        self.disappointment = d.get('disappointment', 0.1)
        self.trust = d.get('trust', 0.8)
        self.positivity = d.get('positivity', 0.5)
    
    def get_negative_average(self):
        """计算负面情绪平均值"""
        return (self.anger + self.anxiety + self.disappointment) / 3
    
    def get_dominant_emotion(self):
        """获取主导情绪"""
        emotions = {
            'anger': self.anger,
            'anxiety': self.anxiety,
            'disappointment': self.disappointment
        }
        
        max_emotion = max(emotions.items(), key=lambda x: x[1])
        if max_emotion[1] > 0.3:
            return max_emotion[0]
        return 'neutral'


def calculate_emotion_similarity(emotion1: Dict, emotion2: Dict) -> float:
    """计算两个情绪向量的相似度"""
    dimensions = ['anger', 'anxiety', 'disappointment', 'trust', 'positivity']
    
    distance = 0
    for dim in dimensions:
        diff = emotion1.get(dim, 0) - emotion2.get(dim, 0)
        distance += diff ** 2

    similarity = 1 / (1 + np.sqrt(distance))
    return similarity


def export_simulation_data(simulator, filepath: str):
    """导出模拟数据供外部分析"""
    data = {
        'metadata': {
            'config': simulator.global_config,
            'start_time': simulator.time_system.start_timestamp,
            'end_time': simulator.time_system.end_timestamp,
            'agent_count': len(simulator.agents),
            'intervention_count': len(simulator.intervention_agents)
        },
        'agents': [
            {
                'aid': agent.aid,
                'type': agent.type,
                'name': agent.name,
                'final_emotion': agent.emotion_vector,
                'behavior_count': len(agent.behavior_history),
                'negative_tendency': agent.negative_behavior_tendency
            }
            for agent in simulator.agents
        ],
        'social_media': simulator.social_media.get_current_info(),
        'group_stats': simulator.group_stats_history,
        'emotion_contagion': simulator.emotion_contagion_events,
        'behavior_patterns': dict(simulator.behavior_patterns)
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"模拟数据已导出至: {filepath}")


def generate_experiment_id():
    """生成唯一的实验ID"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    random_suffix = random.randint(1000, 9999)
    return f"exp_{timestamp}_{random_suffix}"


def merge_analysis_results(result_files: List[str], output_file: str):
    """合并多个分析结果"""
    merged_data = {
        'experiments': [],
        'summary': {
            'total_experiments': len(result_files),
            'avg_max_anger': 0,
            'avg_contagion_events': 0,
            'intervention_effectiveness': []
        }
    }
    
    total_anger = 0
    total_contagion = 0
    
    for filepath in result_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_data['experiments'].append({
                'file': filepath,
                'data': data
            })

            total_anger += data['summary']['emotion_peaks']['max_anger']
            total_contagion += data['summary']['contagion']['total_events']
            
            if 'intervention' in data['summary']:
                merged_data['summary']['intervention_effectiveness'].append(
                    data['summary']['intervention']
                )

    n = len(result_files)
    merged_data['summary']['avg_max_anger'] = total_anger / n
    merged_data['summary']['avg_contagion_events'] = total_contagion / n

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"合并结果已保存至: {output_file}")
    return merged_data


class Event():
    def __init__(self, time, info):
        self.time = time
        self.info = info

    def __lt__(self, other):
        if isinstance(other, Event):
            return self.time < other.time


class Heap():
    def __init__(self):
        self.heap = []

    def __len__(self):
        return len(self.heap)

    def push(self, item):
        heapq.heappush(self.heap, item)

    def pop(self):
        if len(self.heap) <= 0:
            raise Exception("Heap already empty but requires pop.")
        return heapq.heappop(self.heap)

    def top(self):
        return heapq.nsmallest(1, self.heap)[0]

    def empty(self):
        return len(self.heap) == 0

    def print(self):
        print(self.heap)


class SortedList():
    def __init__(self, classType):
        self.loc = []
        self.sorted = []
        self.classType = classType

    def __len__(self):
        return len(self.sorted)

    def __iter__(self):
        return iter(self.sorted)

    def append(self, cid, text):
        self.loc.append(len(self.sorted))
        self.sorted.append(self.classType(cid, text))

    def add_like(self, cid):
        try:
            location = self.loc[cid]
            self.sorted[location].like_num += 1
            self.update_since(location)
        except Exception as e:
            print('Add like fails: %s' % e)

    def update_since(self, location):
        cmt = self.sorted[location]
        cid = cmt.cid
        like_num = cmt.like_num
        pt = location - 1
        while pt >= 0 and self.sorted[pt].like_num < like_num:
            other_cid = self.sorted[pt].cid
            self.loc[other_cid], self.loc[cid] = self.loc[cid], self.loc[other_cid]
            self.sorted[location], self.sorted[pt] = self.sorted[pt], self.sorted[location]
            location = pt
            pt = location - 1

# 定义CST时区（Asia/Shanghai）
cst_tz = pytz.timezone('Asia/Shanghai')

def timestring_to_timestamp(time_string):
    """将CST时间字符串转换为UTC时间戳"""
    naive_dt = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')
    cst_dt = cst_tz.localize(naive_dt)
    return cst_dt.timestamp()

def timestamp_to_timestring(timestamp):
    """将UTC时间戳转换为CST时间字符串"""
    utc_naive_dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
    cst_dt = utc_naive_dt.astimezone(cst_tz)
    return cst_dt.strftime('%Y-%m-%d %H:%M:%S')

def get_sample_prob(f, value_range, inter, num=101):
    delta = (value_range[1] - value_range[0]) / (num - 1)
    x_list = np.linspace(value_range[0], value_range[1], num)
    prob_list = [f(x) / inter for x in x_list]
    cdf_list = [0.0]
    for i in range(1, num):
        cdf_list.append(prob_list[i - 1] * delta + cdf_list[i - 1])
    y = random.random()
    idx = binary_search_insert(cdf_list, y)
    return idx / (num - 1) * (value_range[1] - value_range[0])


def binary_search_insert(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return left


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


class CheckPoint():
    def __init__(self, config, start_time):
        self.config = config
        self.start_time = start_time
        self.checkpoint_time = None
        self.simulator = None
        self.checkpoint_path = config['data_path'].get('save_checkpoint', 'checkpoint/check.pickle')

        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def save_checkpoint(self, simulator):
        """保存检查点"""
        try:
            checkpoint_data = {
                'config': self.config,
                'start_time': self.start_time,
                'checkpoint_time': time.time(),
                'time_system': {
                    'current_time': simulator.time_system.current_time,
                    'priority_queue': self._serialize_priority_queue(simulator.time_system.priority_queue)
                },
                'agents': self._serialize_agents(simulator.agents),
                'intervention_agents': self._serialize_agents(simulator.intervention_agents),
                'social_media': self._serialize_social_media(simulator.social_media),
                'group_stats_history': simulator.group_stats_history,
                'emotion_contagion_events': simulator.emotion_contagion_events,
                'behavior_patterns': dict(simulator.behavior_patterns),
                'experiment_phase': simulator.experiment_phase,
                'phase_start_times': simulator.phase_start_times,
                'contagion_chains': simulator.contagion_chains,
                'agent_emotion_history': dict(simulator.agent_emotion_history)
            }

            write_pickle(checkpoint_data, self.checkpoint_path)
            print(f"[检查点] 保存成功: {self.checkpoint_path}")

        except Exception as e:
            print(f"[检查点] 保存失败: {e}")
            pass

    def load_checkpoint(self):
        """加载检查点"""
        try:
            checkpoint_path = self.config['data_path'].get('load_checkpoint', 'checkpoint/check.pickle')

            if not os.path.exists(checkpoint_path):
                print(f"[检查点] 文件不存在: {checkpoint_path}")
                return False

            checkpoint_data = read_pickle(checkpoint_path)
            print(f"[检查点] 加载成功: {checkpoint_path}")

            from simulator_enhanced import Simulator
            from llm import create_LLM
            from recorder import Recorder

            llm = create_LLM(checkpoint_data['config'])
            recorder = Recorder(checkpoint_data['config'])

            self.simulator = Simulator(checkpoint_data['config'], llm, recorder, self)

            self.simulator.time_system.current_time = checkpoint_data['time_system']['current_time']

            self._restore_priority_queue(
                self.simulator.time_system.priority_queue,
                checkpoint_data['time_system']['priority_queue']
            )

            self._restore_agents(self.simulator.agents, checkpoint_data['agents'])
            self._restore_agents(self.simulator.intervention_agents, checkpoint_data['intervention_agents'])

            self._restore_social_media(self.simulator.social_media, checkpoint_data['social_media'])

            self.simulator.group_stats_history = checkpoint_data['group_stats_history']
            self.simulator.emotion_contagion_events = checkpoint_data['emotion_contagion_events']
            self.simulator.behavior_patterns = defaultdict(int, checkpoint_data['behavior_patterns'])
            self.simulator.experiment_phase = checkpoint_data['experiment_phase']
            self.simulator.phase_start_times = checkpoint_data['phase_start_times']
            self.simulator.contagion_chains = checkpoint_data.get('contagion_chains', [])
            self.simulator.agent_emotion_history = defaultdict(list, checkpoint_data.get('agent_emotion_history', {}))

            print(f"[检查点] 恢复完成，从时间 {timestamp_to_timestring(self.simulator.time_system.current_time)} 继续")
            return True

        except Exception as e:
            print(f"[检查点] 加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _serialize_priority_queue(self, pq):
        """序列化优先队列"""
        events = []
        temp_heap = []
        while not pq.empty():
            event = pq.pop()
            events.append({
                'time': event.time,
                'info': event.info
            })
            temp_heap.append(event)

        for event in temp_heap:
            pq.push(event)

        return events

    def _restore_priority_queue(self, pq, events_data):
        """恢复优先队列，过滤过期事件"""
        current_time = self.simulator.time_system.current_time if self.simulator else 0

        for event_data in events_data:
            event_time = event_data['time']
            if event_time >= current_time:
                pq.push(Event(event_time, event_data['info']))
            else:
                print(f"[检查点] 跳过过期事件: {timestamp_to_timestring(event_time)}")

    def _serialize_agents(self, agents):
        """序列化代理列表"""
        serialized = []
        for agent in agents:
            agent_data = {
                'aid': agent.aid,
                'name': agent.name,
                'type': agent.type,
                'profile': agent.profile,
                'memory': agent.memory,
                'opinion': agent.opinion,
                'social_confidence': agent.social_confidence,
                'emotion': agent.emotion,
                'emotion_vector': agent.emotion_vector.copy() if hasattr(agent, 'emotion_vector') else {},
                'emotion_susceptibility': agent.emotion_susceptibility.copy() if hasattr(agent,
                                                                                         'emotion_susceptibility') else {},
                'behavior_history': agent.behavior_history if hasattr(agent, 'behavior_history') else [],
                'last_active_time': agent.last_active_time if hasattr(agent, 'last_active_time') else 0,
                'negative_behavior_tendency': agent.negative_behavior_tendency if hasattr(agent,
                                                                                          'negative_behavior_tendency') else 0.1
            }

            if hasattr(agent, 'attack_intensity'):
                agent_data['attack_intensity'] = agent.attack_intensity
            if hasattr(agent, 'emotional_manipulation_skill'):
                agent_data['emotional_manipulation_skill'] = agent.emotional_manipulation_skill

            if hasattr(agent, 'intervention_type'):
                agent_data['intervention_type'] = agent.intervention_type
                agent_data['intervention_count'] = agent.intervention_count

            serialized.append(agent_data)

        return serialized

    def _restore_agents(self, agents, agents_data):
        """恢复代理状态"""
        for i, agent_data in enumerate(agents_data):
            if i < len(agents):
                agent = agents[i]
                agent.memory = agent_data['memory']
                agent.opinion = agent_data['opinion']
                agent.social_confidence = agent_data['social_confidence']
                agent.emotion = agent_data['emotion']

                if 'emotion_vector' in agent_data:
                    agent.emotion_vector = agent_data['emotion_vector']
                if 'emotion_susceptibility' in agent_data:
                    agent.emotion_susceptibility = agent_data['emotion_susceptibility']
                if 'behavior_history' in agent_data:
                    agent.behavior_history = agent_data['behavior_history']
                if 'last_active_time' in agent_data:
                    agent.last_active_time = agent_data['last_active_time']
                if 'negative_behavior_tendency' in agent_data:
                    agent.negative_behavior_tendency = agent_data['negative_behavior_tendency']

                if 'attack_intensity' in agent_data:
                    agent.attack_intensity = agent_data['attack_intensity']
                if 'emotional_manipulation_skill' in agent_data:
                    agent.emotional_manipulation_skill = agent_data['emotional_manipulation_skill']
                if 'intervention_count' in agent_data:
                    agent.intervention_count = agent_data['intervention_count']

    def _serialize_social_media(self, social_media):
        """序列化社交媒体状态"""
        def serialize_reply(reply):
            if isinstance(reply, str):
                return {
                    'rid': None,
                    'cid': None,
                    'reply_text': reply,
                    'author_id': None,
                    'timestamp': None,
                    'emotion_scores': {},
                    'is_negative': False,
                    'is_intervention': False,
                    'visibility_weight': 1.0,
                    'filtered': False
                }
            return {
                'rid': getattr(reply, 'rid', None),
                'cid': getattr(reply, 'cid', None),
                'reply_text': getattr(reply, 'reply_text', str(reply)),
                'author_id': getattr(reply, 'author_id', None),
                'timestamp': getattr(reply, 'timestamp', None),
                'emotion_scores': getattr(reply, 'emotion_scores', {}).copy(),
                'is_negative': getattr(reply, 'is_negative', False),
                'is_intervention': getattr(reply, 'is_intervention', False),
                'visibility_weight': getattr(reply, 'visibility_weight', 1.0),
                'filtered': getattr(reply, 'filtered', False)
            }

        def serialize_comment(comment):
            return {
                'cid': comment.cid,
                'comment_text': comment.comment_text,
                'author_id': comment.author_id,
                'like_num': comment.like_num,
                'timestamp': getattr(comment, 'timestamp', None),
                'emotion_scores': getattr(comment, 'emotion_scores', {}).copy(),
                'visibility_weight': getattr(comment, 'visibility_weight', 1.0),
                'is_negative': getattr(comment, 'is_negative', False),
                'is_intervention': getattr(comment, 'is_intervention', False),
                'reply_list': [serialize_reply(reply) for reply in getattr(comment, 'reply_list', [])]
            }

        def serialize_filtered_item(item):
            if isinstance(item, dict):
                copied = item.copy()
                if 'comment' in copied:
                    copied['comment'] = serialize_comment(copied['comment'])
                if 'reply' in copied:
                    copied['reply'] = serialize_reply(copied['reply'])
                return copied
            return {
                'comment': serialize_comment(item),
                'reason': '',
                'timestamp': getattr(item, 'timestamp', None)
            }

        return {
            'view_num': social_media.view_num,
            'like_num': social_media.like_num,
            'repost_num': social_media.repost_num,
            'content_stats': social_media.content_stats.copy(),
            'emotion_history': social_media.emotion_history.copy(),
            'comments': [serialize_comment(comment) for comment in social_media.comment_list.sorted],
            'filtered_comments': [serialize_filtered_item(item) for item in social_media.filtered_comments],
            'detect_list': list(social_media.detect_list),
            'interaction_network': {agent_id: list(targets) for agent_id, targets in social_media.interaction_network.items()}
        }

    def _restore_social_media(self, social_media, media_data):
        """恢复社交媒体状态"""
        from social_media_enhanced import Reply

        def restore_reply(reply_data, fallback_cid=None, fallback_rid=0):
            if isinstance(reply_data, str):
                return reply_data
            reply = Reply(
                reply_data.get('rid') if reply_data.get('rid') is not None else fallback_rid,
                reply_data.get('cid') if reply_data.get('cid') is not None else fallback_cid,
                reply_data.get('reply_text', ''),
                reply_data.get('author_id')
            )
            if reply_data.get('timestamp') is not None:
                reply.timestamp = reply_data['timestamp']
            reply.emotion_scores = reply_data.get('emotion_scores', {})
            reply.is_negative = reply_data.get('is_negative', False)
            reply.is_intervention = reply_data.get('is_intervention', False)
            reply.visibility_weight = reply_data.get('visibility_weight', 1.0)
            reply.filtered = reply_data.get('filtered', False)
            return reply

        def apply_comment_data(comment, comment_data):
            comment.author_id = comment_data.get('author_id')
            comment.like_num = comment_data.get('like_num', 0)
            comment.timestamp = comment_data.get('timestamp', getattr(comment, 'timestamp', None))
            comment.emotion_scores = comment_data.get('emotion_scores', {})
            comment.visibility_weight = comment_data.get('visibility_weight', 1.0)
            comment.is_negative = comment_data.get('is_negative', False)
            comment.is_intervention = comment_data.get('is_intervention', False)
            comment.reply_list = []
            for reply_data in comment_data.get('reply_list', []):
                comment.reply_list.append(
                    restore_reply(reply_data, fallback_cid=comment.cid, fallback_rid=len(comment.reply_list)))

        def restore_comment(comment_data):
            comment = social_media.comment_list.classType(
                comment_data.get('cid', len(social_media.comment_list.sorted)),
                comment_data.get('comment_text', '')
            )
            apply_comment_data(comment, comment_data)
            return comment

        social_media.view_num = media_data['view_num']
        social_media.like_num = media_data['like_num']
        social_media.repost_num = media_data['repost_num']
        social_media.content_stats = media_data['content_stats']
        social_media.emotion_history = media_data['emotion_history']
        social_media.content_stats.setdefault('downranked_comments', 0)

        social_media.comment_list = SortedList(social_media.comment_list.classType)
        restored_comments = [restore_comment(comment_data) for comment_data in media_data.get('comments', [])]
        social_media.comment_list.sorted = restored_comments
        max_cid = max((comment.cid for comment in restored_comments), default=-1)
        social_media.comment_list.loc = [None] * (max_cid + 1)
        for index, comment in enumerate(restored_comments):
            if comment.cid >= 0:
                social_media.comment_list.loc[comment.cid] = index

        social_media.filtered_comments = []
        for item in media_data.get('filtered_comments', []):
            if isinstance(item, dict) and 'comment' in item and isinstance(item['comment'], dict):
                restored_item = item.copy()
                restored_item['comment'] = restore_comment(item['comment'])
                social_media.filtered_comments.append(restored_item)
            elif isinstance(item, dict) and 'reply' in item and isinstance(item['reply'], dict):
                restored_item = item.copy()
                restored_item['reply'] = restore_reply(
                    item['reply'],
                    fallback_cid=item.get('comment_id'),
                    fallback_rid=item['reply'].get('rid', 0)
                )
                social_media.filtered_comments.append(restored_item)
            else:
                social_media.filtered_comments.append(item)

        social_media.detect_list = list(media_data.get('detect_list', []))
        social_media.interaction_network = defaultdict(list, media_data.get('interaction_network', {}))
