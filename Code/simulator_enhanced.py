import re
import time
import numpy as np
from collections import defaultdict
from datetime import datetime

from mechanism import *
from utils_enhanced import *
from attacker_enhanced import SocialAttacker, TrollingAttacker, FactAttacker, EmotionalAttacker
from agent_enhanced import Agent
from intervention_agent import (InterventionAgent, RumorRefutationAgent,
                                EmotionSoothingAgent, RationalGuidanceAgent)
from social_media_enhanced import SocialMedia, Comment, Reply
from emotion_analyzer import EmotionAnalyzer


class TimeSystem():
    def __init__(self, config):
        self.global_config = config
        self.config = config['time_config']
        self.start_timestamp = timestring_to_timestamp(config['time_config']['start_time'])
        self.end_timestamp = timestring_to_timestamp(config['time_config']['end_time'])
        self.current_time = self.start_timestamp

        self.priority_queue = Heap()

    def is_finish(self):
        if self.priority_queue.empty():
            return True
        if self.priority_queue.top().time > self.end_timestamp:
            return True
        return False

    def add_event(self, time, info):
        self.priority_queue.push(Event(time, info))

    def execute_event(self):
        event = self.priority_queue.pop()
        self.current_time = event.time
        return event.time, event.info


class Simulator():
    """
    增强版社交网络模拟器：
    管理多智能体交互、情绪传染、干预策略和实验阶段控制
    """
    def __init__(self, config, llm, recorder, checkpoint):
        self.global_config = config
        self.config = config['simulation_config']
        self.llm = llm
        self.recorder = recorder
        self.checkpoint = checkpoint

        self.time_system = TimeSystem(config)
        self.social_media = SocialMedia(config, llm)
        self.emotion_analyzer = EmotionAnalyzer(llm)

        self.agents = []
        self.intervention_agents = []

        self.group_stats_history = []
        self.emotion_contagion_events = []
        self.behavior_patterns = defaultdict(int)

        self.contagion_chains = []
        self.agent_emotion_history = defaultdict(list)

        self.experiment_phase = 'baseline'
        self.phase_start_times = {}

        self.total_duration = self.time_system.end_timestamp - self.time_system.start_timestamp
        self.phase_times = {
            'baseline_end': self.time_system.start_timestamp + self.total_duration *
                            self.global_config['simulation_config']['experiment_phases']['baseline_duration'],
            'intervention_end': self.time_system.start_timestamp + self.total_duration * (
                    self.global_config['simulation_config']['experiment_phases']['baseline_duration'] +
                    self.global_config['simulation_config']['experiment_phases']['intervention_duration'])
        }

        print(f"配置的基线阶段比例: {self.global_config['simulation_config']['experiment_phases']['baseline_duration']}")
        print(f"配置的干预阶段比例: {self.global_config['simulation_config']['experiment_phases']['intervention_duration']}")

        self.intervention_exit_status = {
            'rumor_refutation': {'should_exit': False, 'exit_time': 0},
            'emotion_soothing': {'should_exit': False, 'exit_time': 0},
            'rational_guidance': {'should_exit': False, 'exit_time': 0}
        }

        self.initialize()


    def get_agent_by_id(self, agent_id):
        for agent in self.agents:
            if hasattr(agent, 'aid') and agent.aid == agent_id:
                return agent

        for agent in self.intervention_agents:
            if hasattr(agent, 'aid') and agent.aid == agent_id:
                return agent

        print(f"Agent with ID {agent_id} not found")
        return None

    def get_user_influence_weight(self, agent_id):
        """计算用户影响力权重（基于粉丝数、活跃度、历史影响力）"""
        agent = self.get_agent_by_id(agent_id)
        if not agent or not hasattr(agent, 'profile') or not agent.profile:
            print(f"Agent {agent_id} has no profile, using default influence weight 0.5")
            return 0.5

        try:
            follower_count = int(agent.profile.get('followed_num', 0))
        except ValueError:
            follower_count = 0

        max_followers_for_norm = 10000.0
        follower_factor = min(follower_count / max_followers_for_norm, 1.0)

        activity_factor = 0.1

        if hasattr(agent, 'behavior_history'):
            behavior_history_length = len(agent.behavior_history)
            activity_factor = min(behavior_history_length / 100, 0.8) + 0.2

        type_factor = 0.7

        user_type = agent.type

        if user_type == 'S':
            type_factor = 1.1
        elif user_type == 'T':
            type_factor = 1.3
        elif user_type == 'F':
            type_factor = 1.2
        elif user_type == 'E':
            type_factor = 1.5

        weight = 0.5 * follower_factor + 0.3 * activity_factor + 0.2 * type_factor
        return max(0.1, min(weight, 1.0))

    def recover_llm(self, llm):
        self.llm = llm
        self.social_media.recover_llm(llm)
        self.emotion_analyzer = EmotionAnalyzer(llm)
        for agent in self.agents:
            agent.recover_llm(llm)

    def initialize(self):
        """初始化模拟器：创建代理、安排初始事件、记录基线状态"""
        user_list = load_user(self.global_config)
        self.recorder.add_record('META', {'user_list': user_list})

        for index, user in enumerate(user_list):
            agent = eval(AGENT_CLASS[user['type']])(self.global_config, index, user, self.llm,self)
            self.agents.append(agent)

            event_time = agent.get_next_execute_time(self.time_system.start_timestamp,
                                                     self.social_media, self)
            if event_time:
                self.time_system.add_event(event_time, ('agent', index))


        any_intervention_enabled = (
                self.global_config['intervention'].get('rumor_refutation', {}).get('enabled', False) or
                self.global_config['intervention'].get('emotion_soothing', {}).get('enabled', False) or
                self.global_config['intervention'].get('rational_guidance', {}).get('enabled', False)
        )

        if any_intervention_enabled:
            self.init_intervention_agents()

        self.phase_start_times['baseline'] = self.time_system.start_timestamp
        self.collect_group_statistics()

    def init_intervention_agents(self):
        """初始化干预代理：辟谣、情绪安抚、理性引导"""
        intervention_config = self.global_config['intervention']

        if intervention_config.get('rumor_refutation', {}).get('enabled', False):
            count = intervention_config['rumor_refutation'].get('agent_count', 3)
            for i in range(count):
                agent = RumorRefutationAgent(
                    self.global_config,
                    len(self.agents) + i,
                    self.llm
                )
                self.intervention_agents.append(agent)

        if intervention_config.get('emotion_soothing', {}).get('enabled', False):
            count = intervention_config['emotion_soothing'].get('agent_count', 2)
            base_id = len(self.agents) + len(self.intervention_agents)
            for i in range(count):
                agent = EmotionSoothingAgent(
                    self.global_config,
                    base_id + i,
                    self.llm
                )
                self.intervention_agents.append(agent)

        if intervention_config.get('rational_guidance', {}).get('enabled', False):
            count = intervention_config['rational_guidance'].get('agent_count', 3)
            base_id = len(self.agents) + len(self.intervention_agents)
            for i in range(count):
                agent = RationalGuidanceAgent(
                    self.global_config,
                    base_id + i,
                    self.llm
                )
                self.intervention_agents.append(agent)


    def agent_workflow(self, current_time, aid, agent_type='agent'):
        """处理代理工作流：模拟用户在社交平台的完整交互流程"""
        if agent_type == 'agent':
            agent = self.agents[aid]
        else:  # intervention
            agent = next((a for a in self.intervention_agents if a.aid == aid), None)
            if not agent:
                print(f"Warning: Intervention agent {aid} not found")
                return current_time

        print(f'----- {agent.name} ({agent.type}) START at {timestamp_to_timestring(current_time)} -----')

        current_state = 'EntryState'
        time_usage = 0
        comment_page = 0
        current_comment_id = None
        action_list = []

        record_info = {
            'user_info': {
                'aid': agent.aid,
                'name': agent.name,
                'type': agent.type,
                'profile': agent.profile,
                'agent_type': agent_type
            }
        }
        trajectory_list = []

        while current_state != 'EndState' and len(action_list) <= self.config['max_step']:
            trajectory_timestamp = current_time + time_usage
            trajectory_info = {'State': current_state, 'timestamp': trajectory_timestamp}

            visible_comments = self.social_media.get_visible_comments(page=comment_page) if current_state == 'MainState' else []
            view_info = get_view_info(self.social_media, current_state, comment_page, current_comment_id)
            trajectory_info['view_info'] = view_info

            if agent_type == 'agent' and current_state == 'MainState':
                contagion_source = self.process_emotion_contagion(agent, view_info, trajectory_timestamp, visible_comments)
                if contagion_source:
                    trajectory_info['contagion_source'] = contagion_source

            agent_action_info = {'timestamp': trajectory_timestamp}
            if current_state == 'MainState':
                agent_action_info['comment_page'] = comment_page
                agent_action_info['visible_comment_ids'] = self.social_media.get_visible_comment_ids(page=comment_page)
            action, text_response, action_info = agent.take_action(view_info, current_state, agent_action_info)

            trajectory_info['action'] = action
            trajectory_info['text_response'] = text_response
            trajectory_info['action_info'] = action_info
            trajectory_info['action_detail'] = agent_action_info
            action_list.append(action)

            is_intervention = (agent_type == 'intervention')
            current_state = self.social_media.action_response(
                action, text_response, action_info, current_comment_id,
                trajectory_timestamp,
                agent.aid, is_intervention
            )

            if isinstance(action_info, dict):
                generated_scores = action_info.get('generated_content_emotion_scores')
                if generated_scores is not None:
                    agent_action_info['generated_content_emotion_scores'] = generated_scores
                    agent_action_info['generated_content_is_negative'] = action_info.get(
                        'generated_content_is_negative', False
                    )

            self.record_behavior_pattern(agent, action, agent_action_info)

            if action in ['Comment', 'Reply'] and text_response:
                emotion_scores = None
                if isinstance(action_info, dict):
                    emotion_scores = action_info.get('generated_content_emotion_scores')
                if emotion_scores is None:
                    emotion_scores = self.emotion_analyzer.analyze_content_emotion(text_response)
                self.emotion_analyzer.update_agent_emotion(agent, {}, emotion_scores, is_intervention_started=(
                        self.experiment_phase == 'intervention'))

            time_usage += int(len(view_info) / 10)
            trajectory_info['time_consume'] = int(len(view_info) / 10)
            comment_page += int(action == 'More Comments')
            if action == 'Detailed Comment':
                current_comment_id = action_info.get('target_comment_id') if isinstance(action_info, dict) else text_response

            if isinstance(action_info, dict):
                action_info['timestamp'] = trajectory_timestamp
                formatted_time = timestamp_to_timestring(trajectory_timestamp)
                print(f'Action: {action} at {formatted_time} (extra: {action_info.get("extra_response", "")})')
            else:
                print(f'Action: {action_info}')

            trajectory_list.append(trajectory_info)

        record_info['trajectory'] = trajectory_list
        record_info['meta'] = {
            'start_time': timestamp_to_timestring(current_time).__str__(),
            'end_time': timestamp_to_timestring(current_time + time_usage).__str__(),
            'total_consume_time': time_usage,
            'final_emotion_state': agent.emotion_vector.copy() if hasattr(agent, 'emotion_vector') else {}
        }

        self.recorder.add_record('TRAJ', record_info)
        print('Finish this workflow.')

        return current_time + time_usage

    def process_emotion_contagion(self, agent, view_info, current_time, visible_comments=None):
        """【修复5】：改进情绪传染处理，正确记录传染链：根据可见评论更新代理情绪，记录传染事件"""
        if visible_comments is None:
            visible_comments = self.social_media.get_visible_comments(page=0)
        if not visible_comments:
            return None

        cumulative_emotion = defaultdict(float)
        emotion_count = 0
        source_agents = []
        intervention_support = 0.0
        intervention_count = 0

        for comment in visible_comments:
            comment_author_id = self.social_media.get_comment_author_id(comment)
            if comment_author_id is None:
                influence_weight = 0.5
                print(f"Warning: Comment author ID is None, using default weight 0.5")
            else:
                influence_weight = self.get_user_influence_weight(comment_author_id)

            comment_text = comment.comment_text if hasattr(comment, 'comment_text') else ""
            if hasattr(comment, 'emotion_scores') and comment.emotion_scores:
                emotions = comment.emotion_scores
            else:
                emotions = {'anger': 0.0, 'anxiety': 0.0, 'disappointment': 0.0, 'negativity': 0.0}

            for emotion, value in emotions.items():
                if emotion in ['anger', 'anxiety', 'disappointment', 'negativity']:
                    cumulative_emotion[emotion] += value * influence_weight
            emotion_count += 1

            if getattr(comment, 'is_intervention', False):
                intervention_support += (
                    emotions.get('trust', 0.0) + emotions.get('positivity', 0.0)
                ) * 0.5 * influence_weight
                intervention_count += 1

            source_agents.append({
                'author_id': comment_author_id,
                'comment_text': comment_text,
                'emotions': emotions,
                'influence_weight': influence_weight
            })

        if emotion_count > 0:
            for emotion in cumulative_emotion:
                cumulative_emotion[emotion] /= emotion_count

            if intervention_count > 0:
                support_value = intervention_support / intervention_count
                cumulative_emotion['intervention_trust_boost'] = support_value
                cumulative_emotion['intervention_positivity_boost'] = support_value

            pre_emotion = agent.emotion_vector.copy()

            self.emotion_analyzer.update_agent_emotion(agent, cumulative_emotion, None, is_intervention_started=(
                        self.experiment_phase == 'intervention'))

            contagion_strength = self.calculate_contagion_strength(pre_emotion, agent.emotion_vector)

            if contagion_strength > 0.02:
                contagion_event = {
                    'timestamp': current_time,
                    'agent_id': agent.aid,
                    'agent_name': agent.name,
                    'pre_emotion': pre_emotion,
                    'post_emotion': agent.emotion_vector.copy(),
                    'contagion_strength': contagion_strength,
                    'source_emotions': dict(cumulative_emotion),
                    'source_comments': visible_comments[:3]
                }

                self.emotion_contagion_events.append(contagion_event)

                self.build_contagion_chain(agent.aid, source_agents, contagion_event)

                self.agent_emotion_history[agent.aid].append({
                    'timestamp': current_time,
                    'emotion': agent.emotion_vector.copy()
                })

                print(f"[情绪传染] {agent.name} - 强度: {contagion_strength:.3f}")
                return contagion_event

        return None

    def build_contagion_chain(self, target_agent_id, source_agents, contagion_event):
        """构建传染链"""
        chain = {
            'target': target_agent_id,
            'sources': source_agents,
            'timestamp': contagion_event['timestamp'],
            'strength': contagion_event['contagion_strength'],
            'emotion_change': {
                'anger': contagion_event['post_emotion']['anger'] - contagion_event['pre_emotion']['anger'],
                'anxiety': contagion_event['post_emotion']['anxiety'] - contagion_event['pre_emotion']['anxiety'],
                'disappointment': contagion_event['post_emotion']['disappointment'] - contagion_event['pre_emotion'][
                    'disappointment']
            }
        }
        self.contagion_chains.append(chain)

    def extract_comments_from_view(self, view_info):
        """从视图信息中提取评论文本"""
        comments = []

        if '热门评论:' in view_info:
            comment_section = view_info.split('热门评论:')[1]
            import re
            comment_matches = re.findall(r'\[\d+\]\s+([^(\n]+?)(?:\s*\(点赞:|$)', comment_section)
            comments.extend([match.strip() for match in comment_matches if match.strip()])

        return comments

    def calculate_contagion_strength(self, pre_emotion, post_emotion):
        strength = 0.0
        weights = {'anger': 0.4, 'anxiety': 0.3, 'disappointment': 0.3}

        for emotion, weight in weights.items():
            if emotion in pre_emotion and emotion in post_emotion:
                delta = abs(post_emotion[emotion] - pre_emotion[emotion])
                strength += delta * weight

        if hasattr(self, 'global_config') and 'agent_config' in self.global_config and \
                'emotion_config' in self.global_config['agent_config'] and \
                'contagion_strength' in self.global_config['agent_config']['emotion_config']:
            strength *= self.global_config['agent_config']['emotion_config']['contagion_strength']

        return strength

    def record_behavior_pattern(self, agent, action, action_info):
        """记录行为模式"""
        if hasattr(agent, 'emotion_vector'):
            emotion_state = self.categorize_emotion_state(agent.emotion_vector)
            pattern_key = f"{emotion_state}_{action}"
            self.behavior_patterns[pattern_key] += 1

            if action == 'Comment' and 'decision_info' in action_info:
                emotions = action_info.get('generated_content_emotion_scores')
                if emotions is None:
                    extra_response = action_info['decision_info'].get('extra_response', {})
                    response_text = extra_response.get('response', '')
                    if response_text:
                        emotions = self.emotion_analyzer.analyze_content_emotion(response_text)
                if emotions and emotions.get('negativity', 0) > 0.5:
                    self.behavior_patterns['negative_comment'] += 1

    def categorize_emotion_state(self, emotion_vector):
        """将情绪状态分类"""
        if emotion_vector['anger'] > 0.6:
            return 'angry'
        elif emotion_vector['anxiety'] > 0.6:
            return 'anxious'
        elif emotion_vector['disappointment'] > 0.6:
            return 'disappointed'
        elif emotion_vector['trust'] < 0.3:
            return 'distrustful'
        else:
            return 'neutral'

    def collect_group_statistics(self):
        stats = {
            'timestamp': self.time_system.current_time,
            'datetime': datetime.fromtimestamp(self.time_system.current_time).isoformat(),
            'phase': self.experiment_phase,
            'phase_progress': self._calculate_phase_progress(),
            'emotion_averages': defaultdict(float),
            'emotion_distributions': defaultdict(list),
            'high_emotion_counts': defaultdict(int),
            'behavior_summary': dict(self.behavior_patterns),
            'social_media_state': self.social_media.get_current_state(),
            'agent_emotions': {}
        }

        ACTIVE_TIME_WINDOW = 120
        current_time = self.time_system.current_time
        active_agents = [a for a in self.agents if hasattr(a, 'last_active_time') and a.last_active_time > 0 and (current_time - a.last_active_time) < ACTIVE_TIME_WINDOW]
        print(f"当前时间: {current_time}")
        print(f"活跃代理数量: {len(active_agents)}")

        stats_agents = self.agents
        stats['active_agent_count'] = len(active_agents)
        stats['stats_agent_count'] = len(stats_agents)

        for agent in self.agents:
            stats['agent_emotions'][agent.aid] = agent.emotion_vector.copy()

        if stats_agents:
            for agent in stats_agents:
                for emotion, value in agent.emotion_vector.items():
                    stats['emotion_averages'][emotion] += value
                    stats['emotion_distributions'][emotion].append(value)

                    if emotion in ['anger', 'anxiety', 'disappointment'] and value > 0.6:
                        stats['high_emotion_counts'][emotion] += 1
                    elif emotion == 'trust' and value < 0.4:
                        stats['high_emotion_counts']['low_trust'] += 1

            n = len(stats_agents)
            for emotion in stats['emotion_averages']:
                stats['emotion_averages'][emotion] /= n

            stats['emotion_std'] = {}
            for emotion, values in stats['emotion_distributions'].items():
                if values:
                    stats['emotion_std'][emotion] = float(np.std(values))

        recent_contagions = [e for e in self.emotion_contagion_events
                             if e['timestamp'] > self.time_system.current_time - 3600]
        stats['contagion_rate'] = len(recent_contagions) / max(1, len(stats_agents))

        stats['contagion_chains_count'] = len(self.contagion_chains)

        self.group_stats_history.append(stats)
        self.recorder.add_record('GROUP_STATS', stats)

    def check_intervention_triggers(self):
        self._check_phase_transition_by_time()

        if self.experiment_phase == 'post-intervention':
            return

        if not self.group_stats_history:
            return

        current_time = self.time_system.current_time
        if self.experiment_phase == 'baseline' and current_time < self.phase_times['baseline_end']:
            return

        latest_stats = self.group_stats_history[-1]
        config_triggers = self.global_config['simulation_config']['experiment_phases']['intervention_trigger']

        high_influence_emergency = False
        for agent in self.get_high_influence_agents():
            emotion_delta = self.calculate_emotion_change_rate(agent, 300)
            if emotion_delta.get('anger', 0) > 0.3:
                high_influence_emergency = True
                print(f"[紧急干预] 高影响力用户{agent.name}愤怒值快速上升: {emotion_delta['anger']:.2f}/5min")
                break

        anger_exceeds = latest_stats['emotion_averages'].get('anger', 0) > config_triggers['emotion_soothing'][
            'anger_threshold']
        negative_ratio_exceeds = latest_stats['social_media_state']['negative_comment_ratio'] > \
                                 config_triggers['rumor_refutation']['negative_ratio_threshold']

        if latest_stats['emotion_averages'].get('anger', 0) > config_triggers['emotion_soothing']['anger_threshold']:
             print(
                 f"[触发条件] 愤怒度: {latest_stats['emotion_averages'].get('anger', 0):.3f} > {config_triggers['emotion_soothing']['anger_threshold']}")

        if latest_stats['social_media_state']['negative_comment_ratio'] > config_triggers['rumor_refutation']['negative_ratio_threshold']:
             print(
                 f"[触发条件] 负面比例: {latest_stats['social_media_state']['negative_comment_ratio']:.3f} > {config_triggers['rumor_refutation']['negative_ratio_threshold']}")

        all_agents_count = latest_stats.get('stats_agent_count', len(self.agents))
        low_trust_ratio = 0
        if all_agents_count > 0:
            low_trust_ratio = latest_stats['high_emotion_counts'].get('low_trust', 0) / all_agents_count

        low_trust_exceeds = low_trust_ratio > config_triggers['rational_guidance']['low_trust_ratio_threshold']
        print(f"低信任代理数: {latest_stats['high_emotion_counts'].get('low_trust', 0)}")
        print(f"低信任比例: {low_trust_ratio}")

        if self.intervention_agents and self.experiment_phase in ('baseline', 'intervention'):
            trigger_conditions = [negative_ratio_exceeds, anger_exceeds, low_trust_exceeds, high_influence_emergency]
            print(
                f"[触发状态] 负面比例超标: {negative_ratio_exceeds}, 愤怒值超标: {anger_exceeds}, 低信任比例超标: {low_trust_exceeds}, 高影响力用户紧急: {high_influence_emergency}")

            if self.experiment_phase == 'baseline' and any(trigger_conditions):
                self._start_intervention_phase(
                    current_time,
                    negative_ratio_exceeds,
                    anger_exceeds,
                    low_trust_exceeds,
                    high_influence_emergency
                )
            elif self.experiment_phase == 'intervention' and any(trigger_conditions):
                for agent in self.intervention_agents:
                    if high_influence_emergency and isinstance(agent, EmotionSoothingAgent):
                        self._activate_agent(agent, 'high_influence_emergency')
                    if isinstance(agent, RumorRefutationAgent) and negative_ratio_exceeds:
                        self._activate_agent(agent, 'negative_ratio_threshold')
                    if isinstance(agent, EmotionSoothingAgent) and anger_exceeds:
                        self._activate_agent(agent, 'anger_threshold')
                    if isinstance(agent, RationalGuidanceAgent) and low_trust_exceeds:
                        self._activate_agent(agent, 'low_trust_threshold')

            if low_trust_ratio > config_triggers['rational_guidance']['low_trust_ratio_threshold']:
                print(f"[触发条件] 低信任比例: {low_trust_ratio:.3f} > {config_triggers['rational_guidance']['low_trust_ratio_threshold']}")

        if self.experiment_phase == 'intervention' and self.group_stats_history:
            latest_stats = self.group_stats_history[-1]
            current_time = self.time_system.current_time
            exit_conditions = self.global_config['simulation_config']['experiment_phases'].get('exit_conditions', {})

            if exit_conditions.get('rumor_refutation') and not self.intervention_exit_status['rumor_refutation'][
                'should_exit']:
                rumour_exit = exit_conditions['rumor_refutation']
                current_ratio = latest_stats['social_media_state'].get('negative_comment_ratio', 1.0)
                if current_ratio < rumour_exit['negative_ratio_below_threshold']:
                    if self.intervention_exit_status['rumor_refutation']['exit_time'] == 0:
                        self.intervention_exit_status['rumor_refutation']['exit_time'] = current_time
                    elif current_time - self.intervention_exit_status['rumor_refutation']['exit_time'] >= rumour_exit[
                        'sustain_period']:
                        self.intervention_exit_status['rumor_refutation']['should_exit'] = True
                        print("[退出条件满足] 辟谣干预：负面比例已持续达标")
                else:
                    self.intervention_exit_status['rumor_refutation']['exit_time'] = 0

            if exit_conditions.get('emotion_soothing') and not self.intervention_exit_status['emotion_soothing'][
                'should_exit']:
                emotion_exit = exit_conditions['emotion_soothing']
                current_anger = latest_stats['emotion_averages'].get('anger', 1.0)
                if current_anger < emotion_exit['anger_below_threshold']:
                    if self.intervention_exit_status['emotion_soothing']['exit_time'] == 0:
                        self.intervention_exit_status['emotion_soothing']['exit_time'] = current_time
                    elif current_time - self.intervention_exit_status['emotion_soothing']['exit_time'] >= emotion_exit[
                        'sustain_period']:
                        self.intervention_exit_status['emotion_soothing']['should_exit'] = True
                        print("[退出条件满足] 情绪安抚干预：愤怒值已持续达标")
                else:
                    self.intervention_exit_status['emotion_soothing']['exit_time'] = 0

            if exit_conditions.get('rational_guidance') and not self.intervention_exit_status['rational_guidance'][
                'should_exit']:
                rational_exit = exit_conditions['rational_guidance']
                active_count = len(
                    [a for a in self.agents if hasattr(a, 'last_active_time') and a.last_active_time > 0 and
                     (current_time - a.last_active_time) < 600])
                low_trust_count = latest_stats['high_emotion_counts'].get('low_trust', 0)
                low_trust_ratio = low_trust_count / active_count if active_count > 0 else 0

                if low_trust_ratio < rational_exit['low_trust_ratio_below_threshold']:
                    if self.intervention_exit_status['rational_guidance']['exit_time'] == 0:
                        self.intervention_exit_status['rational_guidance']['exit_time'] = current_time
                    elif current_time - self.intervention_exit_status['rational_guidance']['exit_time'] >= \
                            rational_exit['sustain_period']:
                        self.intervention_exit_status['rational_guidance']['should_exit'] = True
                        print("[退出条件满足] 理性引导干预：低信任比例已持续达标")
                else:
                    self.intervention_exit_status['rational_guidance']['exit_time'] = 0

            for agent in self.intervention_agents:
                if hasattr(agent, 'intervention_type') and hasattr(agent, 'active') and agent.active:
                    if self.intervention_exit_status[agent.intervention_type]['should_exit']:
                        agent.active = False
                        agent.force_intervene = False
                        agent.pending_intervention_count = 0
                        agent.trigger_reason = None
                        print(f"[干预停止] {agent.name} 已停止干预")

    def _start_intervention_phase(self, current_time, negative_ratio_exceeds,
                                  anger_exceeds, low_trust_exceeds,
                                  high_influence_emergency):
        if self.experiment_phase != 'baseline':
            return

        print("=== INTERVENTION PHASE TRIGGERED ===")
        self.experiment_phase = 'intervention'
        self.phase_start_times['intervention'] = current_time

        intervention_ratio = self.global_config['simulation_config']['experiment_phases']['intervention_duration']
        intervention_duration = self.total_duration * intervention_ratio
        self.phase_times['intervention_end'] = min(
            self.time_system.end_timestamp,
            current_time + intervention_duration
        )

        for status in self.intervention_exit_status.values():
            status['should_exit'] = False
            status['exit_time'] = 0

        checkpoint_cfg = self.config.get('checkpointing', {})
        if checkpoint_cfg.get('enabled', True) and checkpoint_cfg.get('save_on_phase_transition', True):
            self.save_checkpoint_if_needed(force=True, reason='intervention_phase_start')

        for agent in self.intervention_agents:
            if high_influence_emergency and isinstance(agent, EmotionSoothingAgent):
                self._activate_agent(agent, 'high_influence_emergency')
            if isinstance(agent, RumorRefutationAgent) and negative_ratio_exceeds:
                self._activate_agent(agent, 'negative_ratio_threshold')
            if isinstance(agent, EmotionSoothingAgent) and anger_exceeds:
                self._activate_agent(agent, 'anger_threshold')
            if isinstance(agent, RationalGuidanceAgent) and low_trust_exceeds:
                self._activate_agent(agent, 'low_trust_threshold')

    def _check_phase_transition_by_time(self):
        """根据时间自动转换实验阶段"""
        current_time = self.time_system.current_time

        enable_auto_transition = self.global_config['simulation_config']['experiment_phases'].get('enable_auto_transition', False)
        if not enable_auto_transition:
            return

        if self.experiment_phase == 'intervention' and current_time >= self.phase_times['intervention_end']:
            self.experiment_phase = 'post-intervention'
            self.phase_start_times['post-intervention'] = current_time
            print(f"=== 自动进入干预后阶段 ===")
            checkpoint_cfg = self.config.get('checkpointing', {})
            if checkpoint_cfg.get('enabled', True) and checkpoint_cfg.get('save_on_phase_transition', True):
                self.save_checkpoint_if_needed(force=True, reason='post_intervention_phase_start')

            for agent in self.intervention_agents:
                if hasattr(agent, 'active'):
                    agent.active = False
                    if hasattr(agent, 'force_intervene'):
                        agent.force_intervene = False
                    if hasattr(agent, 'pending_intervention_count'):
                        agent.pending_intervention_count = 0
                    if hasattr(agent, 'trigger_reason'):
                        agent.trigger_reason = None

    def _calculate_phase_progress(self):
        """计算当前阶段的进度百分比"""
        current_time = self.time_system.current_time
        if self.experiment_phase == 'baseline':
            baseline_end = self.phase_start_times.get('intervention', self.phase_times['baseline_end'])
            phase_duration = baseline_end - self.time_system.start_timestamp
            progress = (current_time - self.time_system.start_timestamp) / phase_duration if phase_duration > 0 else 0
        elif self.experiment_phase == 'intervention':
            intervention_start = self.phase_start_times.get('intervention', self.phase_times['baseline_end'])
            phase_duration = self.phase_times['intervention_end'] - intervention_start
            progress = (current_time - intervention_start) / phase_duration if phase_duration > 0 else 0
        else:  # post-intervention
            post_start = self.phase_start_times.get('post-intervention', self.phase_times['intervention_end'])
            phase_duration = self.time_system.end_timestamp - post_start
            progress = (current_time - post_start) / phase_duration if phase_duration > 0 else 0
        return min(1.0, max(0.0, progress))

    def _activate_agent(self, agent, trigger_reason=None):
        was_active = getattr(agent, 'active', False)
        agent.active = True
        if hasattr(agent, 'force_intervene'):
            agent.force_intervene = True
        if hasattr(agent, 'pending_intervention_count'):
            agent.pending_intervention_count = max(1, getattr(agent, 'pending_intervention_count', 0))
        if hasattr(agent, 'trigger_reason'):
            agent.trigger_reason = trigger_reason

        if was_active:
            return

        next_time = agent.get_next_execute_time(
            self.time_system.current_time,
            self.social_media,
            self
        )

        if next_time:
            self.time_system.add_event(next_time, ('intervention', agent.aid))
            print(f"[干预激活] {agent.name} 将在 {timestamp_to_timestring(next_time)} 开始行动")

    def get_high_influence_agents(self):
        """返回影响力排名前20%的用户"""
        if not self.agents:
            return []

        sorted_agents = sorted(
            self.agents,
            key=lambda a: self.get_user_influence_weight(a.aid),
            reverse=True
        )
        top_count = max(1, int(len(sorted_agents) * 0.2))
        return sorted_agents[:top_count]

    def calculate_emotion_change_rate(self, agent, time_window):
        current_time = self.time_system.current_time
        start_time = current_time - time_window

        relevant_entries = [
            entry for entry in agent.emotion_history
            if start_time <= entry['step'] <= current_time
        ]

        if not relevant_entries:
            return {emotion: 0.0 for emotion in agent.emotion_vector}

        earliest_entry = min(relevant_entries, key=lambda x: x['step'])
        earliest_emotions = earliest_entry['emotion_vector']

        emotion_deltas = {}
        for emotion, current_value in agent.emotion_vector.items():
            past_value = earliest_emotions.get(emotion, 0.0)
            emotion_deltas[emotion] = current_value - past_value

        return emotion_deltas

    def save_checkpoint_if_needed(self, event_count=None, force=False, reason='periodic'):
        if not hasattr(self, 'checkpoint') or self.checkpoint is None:
            return

        checkpoint_cfg = self.config.get('checkpointing', {})
        if not checkpoint_cfg.get('enabled', True):
            return

        if force:
            self.checkpoint.save_checkpoint(self)
            return

        if event_count is None:
            return

        save_every = max(1, int(checkpoint_cfg.get('save_every_n_events', 50)))
        if event_count > 0 and event_count % save_every == 0:
            self.checkpoint.save_checkpoint(self)

    def run(self):
        if hasattr(self, 'checkpoint') and hasattr(self.checkpoint, 'simulator'):
            self.fix_time_system_after_checkpoint()

        """【修复8】：统计采样频率"""
        event_count = 0
        last_stats_time = self.time_system.start_timestamp

        stats_interval = self.total_duration / 80

        while not self.time_system.is_finish():
            current_time, event_info = self.time_system.execute_event()

            if current_time - last_stats_time >= stats_interval:
                self.collect_group_statistics()
                self.check_intervention_triggers()
                last_stats_time = current_time

            if isinstance(event_info, tuple):
                event_type, agent_id = event_info
                end_time = self.agent_workflow(current_time, agent_id, event_type)

                if event_type == 'agent':
                    agent = self.agents[agent_id]
                    agent.step(current_time)
                else:
                    agent = next((a for a in self.intervention_agents if a.aid == agent_id), None)

                if agent:
                    next_time = agent.get_next_execute_time(end_time, self.social_media, self)
                    if next_time:
                        self.time_system.add_event(next_time, (event_type, agent_id))
            else:
                aid = event_info
                agent = self.agents[aid]
                agent.step(current_time)
                end_time = self.agent_workflow(current_time, aid, 'agent')
                event_time = self.agents[aid].get_next_execute_time(end_time, self.social_media, self)
                if event_time:
                    self.time_system.add_event(event_time, ('agent', aid))

            if event_count % 5 == 0:
                print(f'### Event {event_count} completed ###')
            event_count += 1
            self.save_checkpoint_if_needed(event_count=event_count)

        self.collect_group_statistics()
        self.save_checkpoint_if_needed(force=True, reason='simulation_end')
        self.finalize_simulation()

    def fix_time_system_after_checkpoint(self):
        current_time = self.time_system.current_time
        print(f"[时间修复] 当前时间: {timestamp_to_timestring(current_time)}")

        valid_events = []
        while not self.time_system.priority_queue.empty():
            event = self.time_system.priority_queue.pop()
            if event.time >= current_time:
                valid_events.append(event)

        for event in valid_events:
            self.time_system.priority_queue.push(event)

        if not valid_events:
            print("[时间修复] 重新安排所有代理")
            for agent in self.agents:
                next_time = agent.get_next_execute_time(current_time, self.social_media, self)
                if next_time:
                    self.time_system.add_event(next_time, ('agent', agent.aid))

    def finalize_simulation(self):
        """完成模拟并生成最终报告"""
        self.recorder.add_record('META', {
            'final_social_media': self.social_media.get_current_info(),
            'experiment_phases': self.phase_start_times,
            'total_events': len(self.emotion_contagion_events),
            'behavior_patterns': dict(self.behavior_patterns),
            'contagion_chains': self.contagion_chains
        })

        summary = self.generate_summary_report()
        self.recorder.add_record('SUMMARY', summary)

        print("\n=== SIMULATION COMPLETED ===")
        print(f"Total emotion contagion events: {len(self.emotion_contagion_events)}")
        print(f"Contagion chains formed: {len(self.contagion_chains)}")
        print(f"Final negative comment ratio: {summary['final_negative_ratio']:.2%}")
        print(f"Maximum anger level reached: {summary['max_anger']:.2f}")

    def generate_summary_report(self):
        max_anger = max((s['emotion_averages'].get('anger', 0) for s in self.group_stats_history), default=0)
        max_anxiety = max((s['emotion_averages'].get('anxiety', 0) for s in self.group_stats_history), default=0)

        final_stats = self.group_stats_history[-1] if self.group_stats_history else {}

        intervention_effect = None
        if 'intervention' in self.phase_start_times:
            pre_intervention = [s for s in self.group_stats_history
                                if s['timestamp'] < self.phase_start_times['intervention']]
            intervention_data = [s for s in self.group_stats_history
                                 if self.phase_start_times['intervention'] <= s['timestamp'] < self.phase_times[
                                     'intervention_end']]
            post_start = self.phase_start_times.get('post-intervention', self.phase_times.get('intervention_end'))
            post_intervention = [
                s for s in self.group_stats_history
                if post_start is not None and s['timestamp'] >= post_start
            ]

            comparison_data = post_intervention if post_intervention else intervention_data
            comparison_window = 'post-intervention' if post_intervention else 'intervention'

            if pre_intervention and comparison_data:
                pre_data_points = min(10, len(pre_intervention))
                pre_anger_data = [s['emotion_averages'].get('anger', 0) for s in pre_intervention[-pre_data_points:]]
                print(f"干预前最后{pre_data_points}个时间点的愤怒情绪值:", pre_anger_data)
                pre_anger = np.mean(pre_anger_data)

                post_data_points = min(10, len(comparison_data))
                post_anger_data = [s['emotion_averages'].get('anger', 0) for s in comparison_data[-post_data_points:]]
                print(f"{comparison_window}最后{post_data_points}个时间点的愤怒情绪值:", post_anger_data)
                post_anger = np.mean(post_anger_data)
                anger_change = pre_anger - post_anger

                intervention_phase_effect = {}
                if intervention_data:
                    phase_data_points = min(10, len(intervention_data))
                    phase_start_anger_data = [
                        s['emotion_averages'].get('anger', 0)
                        for s in intervention_data[:phase_data_points]
                    ]
                    phase_end_anger_data = [
                        s['emotion_averages'].get('anger', 0)
                        for s in intervention_data[-phase_data_points:]
                    ]
                    phase_start_anger = np.mean(phase_start_anger_data)
                    phase_end_anger = np.mean(phase_end_anger_data)
                    phase_anger_change = phase_start_anger - phase_end_anger
                    intervention_phase_effect = {
                        'intervention_start_anger_avg': phase_start_anger,
                        'intervention_end_anger_avg': phase_end_anger,
                        'intervention_phase_anger_reduction': phase_anger_change,
                        'intervention_phase_anger_reduction_percentage': (
                            phase_anger_change / phase_start_anger * 100
                        ) if phase_start_anger > 0 else 0,
                        'intervention_phase_window_points': phase_data_points
                    }

                intervention_effect = {
                    'anger_reduction': anger_change,
                    'anger_reduction_percentage': (anger_change / pre_anger * 100) if pre_anger > 0 else 0,
                    'intervention_comments': self.social_media.content_stats['intervention_comments'],
                    'filtered_comments': self.social_media.content_stats['filtered_comments'],
                    'pre_anger_avg': pre_anger,
                    'post_anger_avg': post_anger,
                    'comparison_window': comparison_window
                }
                intervention_effect.update(intervention_phase_effect)

        return {
            'max_anger': max_anger,
            'max_anxiety': max_anxiety,
            'final_negative_ratio': final_stats.get('social_media_state', {}).get('negative_comment_ratio', 0),
            'total_comments': self.social_media.content_stats['total_comments'],
            'emotion_contagion_events': len(self.emotion_contagion_events),
            'contagion_chains': len(self.contagion_chains),
            'intervention_effect': intervention_effect,
            'behavior_patterns': dict(self.behavior_patterns),
            'final_emotion_averages': dict(final_stats.get('emotion_averages', {}))
        }

    def get_analysis_data(self):
        """获取用于分析的数据"""
        return {
            'group_stats_history': self.group_stats_history,
            'emotion_contagion_events': self.emotion_contagion_events,
            'contagion_chains': self.contagion_chains,
            'behavior_patterns': dict(self.behavior_patterns),
            'social_media_emotion_history': self.social_media.emotion_history,
            'interaction_network': dict(self.social_media.interaction_network),
            'summary': self.generate_summary_report()
        }
