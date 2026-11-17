import random
import time
import re
import numpy as np
from agent_enhanced import Agent
from mechanism import get_action_info
from utils_enhanced import timestring_to_timestamp


class EmojiEmbedding:
    """表情符号情绪嵌入系统：将表情符号和标点符号映射为情绪向量"""

    def __init__(self):
        self.emoji_vectors = {
            # 愤怒相关
            '😠': np.array([0.9, 0.3, 0.2, 0.1, 0.1]),
            '😡': np.array([1.0, 0.2, 0.3, 0.0, 0.0]),
            '🤬': np.array([1.0, 0.4, 0.3, 0.0, 0.0]),
            '💢': np.array([0.8, 0.5, 0.2, 0.1, 0.1]),

            # 焦虑相关
            '😰': np.array([0.2, 0.9, 0.3, 0.2, 0.1]),
            '😨': np.array([0.3, 1.0, 0.2, 0.1, 0.0]),
            '😱': np.array([0.4, 0.9, 0.3, 0.1, 0.0]),
            '😖': np.array([0.3, 0.8, 0.5, 0.2, 0.1]),

            # 失望相关
            '😔': np.array([0.1, 0.3, 0.9, 0.3, 0.2]),
            '😞': np.array([0.2, 0.2, 1.0, 0.2, 0.1]),
            '😟': np.array([0.2, 0.5, 0.8, 0.3, 0.2]),
            '💔': np.array([0.3, 0.3, 0.9, 0.1, 0.0]),

            # 积极相关
            '😊': np.array([0.0, 0.0, 0.0, 0.9, 1.0]),
            '😄': np.array([0.0, 0.0, 0.0, 0.8, 0.9]),
            '❤️': np.array([0.0, 0.0, 0.0, 1.0, 0.9]),
            '👍': np.array([0.0, 0.0, 0.0, 0.8, 0.8]),

            # 中性
            '😐': np.array([0.3, 0.3, 0.3, 0.5, 0.5]),
            '🤔': np.array([0.2, 0.4, 0.2, 0.6, 0.6])
        }

        self.punctuation_intensity = {
            '!!!': 0.8,
            '！！！': 0.8,
            '???': 0.7,
            '？？？': 0.7,
            '...': 0.5,
            '。。。': 0.5,
            '~~': 0.3,
            '～～': 0.3
        }

    def extract_emotion_from_emojis(self, text):
        """从文本中提取表情符号的情绪向量并归一化"""
        emotion_vector = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
        emoji_count = 0

        for emoji, vector in self.emoji_vectors.items():
            if emoji in text:
                count = text.count(emoji)
                emotion_vector += vector * count
                emoji_count += count

        if emoji_count > 0:
            emotion_vector /= emoji_count
            emotion_vector = np.clip(emotion_vector, 0, 1)

        return emotion_vector

    def get_punctuation_intensity(self, text):
        """获取标点符号表达的情绪强度"""
        max_intensity = 0.0

        for pattern, intensity in self.punctuation_intensity.items():
            if pattern in text:
                max_intensity = max(max_intensity, intensity)

        return max_intensity


class SemanticEmotionAnalyzer:
    """基于语义的情绪分析器：结合词向量、表情符号和标点符号分析文本情绪"""

    def __init__(self, llm):
        self.llm = llm
        self.emoji_embedding = EmojiEmbedding()

        self.emotion_embeddings = {
            'anger': np.array([1.0, 0.2, 0.3, 0.1, 0.0]),
            'anxiety': np.array([0.2, 1.0, 0.3, 0.2, 0.1]),
            'disappointment': np.array([0.3, 0.3, 1.0, 0.2, 0.1]),
            'trust': np.array([0.1, 0.1, 0.1, 1.0, 0.8]),
            'positivity': np.array([0.0, 0.0, 0.1, 0.8, 1.0])
        }

        self.word_vectors = {
            # 愤怒相关词汇
            '愤怒': np.array([0.9, 0.2, 0.2, 0.1, 0.1]),
            '生气': np.array([0.8, 0.2, 0.2, 0.2, 0.2]),
            '恼火': np.array([0.7, 0.3, 0.2, 0.2, 0.2]),
            '暴怒': np.array([1.0, 0.3, 0.2, 0.0, 0.0]),
            '气死': np.array([0.9, 0.4, 0.3, 0.1, 0.0]),
            '可恶': np.array([0.8, 0.2, 0.4, 0.1, 0.1]),

            # 焦虑相关词汇
            '焦虑': np.array([0.2, 0.9, 0.3, 0.2, 0.1]),
            '担心': np.array([0.1, 0.8, 0.3, 0.3, 0.2]),
            '害怕': np.array([0.3, 0.9, 0.2, 0.1, 0.1]),
            '恐慌': np.array([0.4, 1.0, 0.3, 0.0, 0.0]),
            '不安': np.array([0.2, 0.7, 0.4, 0.3, 0.2]),

            # 失望相关词汇
            '失望': np.array([0.2, 0.2, 0.9, 0.2, 0.1]),
            '沮丧': np.array([0.2, 0.3, 0.8, 0.2, 0.2]),
            '绝望': np.array([0.3, 0.4, 1.0, 0.0, 0.0]),
            '无奈': np.array([0.1, 0.3, 0.7, 0.3, 0.2]),
            '糟糕': np.array([0.3, 0.4, 0.6, 0.2, 0.1]),

            # 积极词汇
            '开心': np.array([0.0, 0.0, 0.0, 0.8, 0.9]),
            '快乐': np.array([0.0, 0.0, 0.0, 0.7, 1.0]),
            '希望': np.array([0.0, 0.1, 0.0, 0.8, 0.8]),
            '相信': np.array([0.0, 0.0, 0.0, 0.9, 0.7]),
            '支持': np.array([0.0, 0.0, 0.0, 0.8, 0.8])
        }

    def analyze_text_emotion(self, text):
        """综合词向量、表情符号和标点符号分析文本情绪"""
        word_emotion = self._analyze_with_word_vectors(text)

        emoji_emotion = self.emoji_embedding.extract_emotion_from_emojis(text)

        punctuation_intensity = self.emoji_embedding.get_punctuation_intensity(text)

        combined_emotion = word_emotion * 0.6 + emoji_emotion * 0.3 + punctuation_intensity * 0.1

        if punctuation_intensity > 0.5:
            combined_emotion[:3] *= (1 + punctuation_intensity * 0.3)

        combined_emotion = np.clip(combined_emotion, 0, 1)

        return {
            'anger': float(combined_emotion[0]),
            'anxiety': float(combined_emotion[1]),
            'disappointment': float(combined_emotion[2]),
            'trust': float(combined_emotion[3]),
            'positivity': float(combined_emotion[4]),
            'overall_negativity': float(np.mean(combined_emotion[:3])),
            'punctuation_intensity': punctuation_intensity
        }

    def _analyze_with_word_vectors(self, text):
        """使用词向量分析文本"""
        emotion_vector = np.array([0.1, 0.1, 0.1, 0.5, 0.5])
        word_count = 0

        for word, vector in self.word_vectors.items():
            if word in text:
                count = text.count(word)
                emotion_vector += vector * count
                word_count += count

        if word_count > 0:
            emotion_vector /= word_count

        return emotion_vector

    def calculate_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)

    def analyze(self, context):
        """分析文本情绪并返回标准化分数（0-1）"""
        emotion_scores = {
            'anger': 0.0,
            'anxiety': 0.0,
            'disappointment': 0.0
        }

        anger_keywords = ['愤怒', '生气', '讨厌', '可恶']
        anxiety_keywords = ['担心', '害怕', '焦虑', '紧张']
        disappointment_keywords = ['失望', '遗憾', '沮丧', '难过']

        for keyword in anger_keywords:
            if keyword in context:
                emotion_scores['anger'] += 0.3

        for keyword in anxiety_keywords:
            if keyword in context:
                emotion_scores['anxiety'] += 0.3

        for keyword in disappointment_keywords:
            if keyword in context:
                emotion_scores['disappointment'] += 0.3

        for emotion in emotion_scores:
            emotion_scores[emotion] = min(1.0, max(0.0, emotion_scores[emotion]))

        return emotion_scores

class InterventionAgent(Agent):
    """
    干预代理：通过语义情绪分析检测负面趋势，采用多样化策略生成干预内容，缓解社交平台负面情绪
    """

    def __init__(self, config, aid, feature, llm):
        intervention_type = feature.get('intervention_type', 'rumor_refutation')

        name_map = {
            'rumor_refutation': f"辟谣专家{aid}",
            'emotion_soothing': f"情绪安抚者{aid}",
            'rational_guidance': f"理性引导者{aid}"
        }

        intervention_feature = {
            'name': name_map.get(intervention_type, f"干预者{aid}"),
            'type': 'intervention',
            'profile': {
                'uid': f"INT_{aid}",
                'followed_num': str(random.randint(5000, 10000)),
                'following_num': str(random.randint(100, 500)),
                'description': feature.get('description', '...')
            }
        }

        super().__init__(config, aid, intervention_feature, llm)

        self.intervention_type = feature.get('intervention_type', 'rumor_refutation')
        self.intervention_config = config.get('intervention', {})
        self.runtime_config = self.intervention_config.get('runtime', {})

        self.target_emotions = config.get('target_emotions', ['anger', 'anxiety', 'disappointment'])
        self.strategy_effectiveness = {}
        self.intervention_history = []

        self.semantic_analyzer = SemanticEmotionAnalyzer(llm)

        self.emotion_vector = {
            'anger': 0.02,
            'anxiety': 0.02,
            'disappointment': 0.02,
            'trust': 0.98,
            'positivity': 0.95
        }

        self.emotion_susceptibility = {
            'anger': 0.05,
            'anxiety': 0.05,
            'disappointment': 0.05,
            'trust': 3.0
        }

        self.active = False
        self.force_intervene = False
        self.pending_intervention_count = 0
        self.trigger_reason = None
        self.forced_strategy = None

        self.intervention_count = 0
        self.max_interventions_per_period = 10
        self.last_intervention_time = None
        self.intervention_cooldown = 120
        self.consecutive_interventions = 0
        self.intervention_effectiveness = 0.0

        self.negative_emotion_threshold = 0.25
        self.misinformation_threshold = 0.3

        self.intervention_strategies = {
            'rumor_refutation': ['fact_check', 'source_request', 'logical_analysis'],
            'emotion_soothing': ['empathy', 'perspective', 'hope'],
            'rational_guidance': ['question', 'analysis', 'solution']
        }

        self.used_strategies = []

    def should_intervene(self, social_media_state, current_time):
        """判断是否需要干预：基于情绪强度、趋势和冷却时间"""
        emotion_trend = self._calculate_emotion_trend(social_media_state, window_size=5)
        dominant_emotion, intensity = self._get_dominant_emotion(social_media_state)

        if self.last_intervention_time is not None and current_time - self.last_intervention_time < self.intervention_cooldown / 3:
            return False

        if self.intervention_count >= self.max_interventions_per_period * 3:
            return False

        if self.intervention_type == 'rumor_refutation':
            return self._detect_misinformation_risk(social_media_state)
        elif self.intervention_type == 'emotion_soothing':

            emotion_condition = (
                    (dominant_emotion in self.target_emotions and intensity > 0.5) or
                    (dominant_emotion in ['anger', 'anxiety', 'disappointment'] and emotion_trend > 0.1)
            )
            return emotion_condition or self._detect_high_emotion(social_media_state)
        elif self.intervention_type == 'rational_guidance':

            return (
                    self._detect_irrational_discussion(social_media_state) or
                    (dominant_emotion in self.target_emotions and emotion_trend > 0.1)
            )

        return (
                (dominant_emotion in self.target_emotions and intensity > 0.5) or
                (dominant_emotion in ['anger', 'anxiety', 'disappointment'] and emotion_trend > 0.1)
        )



    def _get_dominant_emotion(self, social_media_state):
        """分析社交媒体状态，返回主导情绪及其强度"""
        if not social_media_state:
            return 'neutral', 0.0

        if 'users' not in social_media_state:
            emotion_scores = {'anger': 0.0, 'anxiety': 0.0, 'disappointment': 0.0, 'trust': 0.0, 'positivity': 0.0}
            score_list = social_media_state.get('emotion_scores', [])
            if score_list:
                for score in score_list:
                    for emotion in emotion_scores:
                        emotion_scores[emotion] += score.get(emotion, 0.0)
                for emotion in emotion_scores:
                    emotion_scores[emotion] /= len(score_list)
            else:
                overall_emotion = social_media_state.get('overall_emotion', {})
                for emotion in emotion_scores:
                    emotion_scores[emotion] = overall_emotion.get(emotion, 0.0)

            dominant_emotion = max(emotion_scores, key=lambda k: emotion_scores[k])
            return dominant_emotion, emotion_scores[dominant_emotion]

        emotion_scores = {'anger': 0.0, 'anxiety': 0.0, 'disappointment': 0.0, 'trust': 0.0, 'positivity': 0.0}
        user_count = 0

        for user_id, user_data in social_media_state['users'].items():
            if 'emotion_vector' in user_data:
                user_count += 1
                for emotion, score in user_data['emotion_vector'].items():
                    if emotion in emotion_scores:
                        emotion_scores[emotion] += score

        if user_count > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= user_count

        dominant_emotion = max(emotion_scores, key=lambda k: emotion_scores[k])
        return dominant_emotion, emotion_scores[dominant_emotion]

    def _calculate_emotion_trend(self, social_media_state, window_size=5):
        """计算情绪变化趋势（最近window_size步的情绪变化率）"""
        if not hasattr(self, 'emotion_history'):
            self.emotion_history = []
            return 0.0

        current_emotion, _ = self._get_dominant_emotion(social_media_state)
        self.emotion_history.append(current_emotion)

        if len(self.emotion_history) > window_size:
            self.emotion_history = self.emotion_history[-window_size:]

        if len(self.emotion_history) < 2:
            return 0.0

        emotion_values = {
            'anger': 0.8, 'anxiety': 0.7, 'disappointment': 0.6,
            'trust': 0.3, 'positivity': 0.2, 'neutral': 0.5
        }

        changes = []
        for i in range(1, len(self.emotion_history)):
            prev_val = emotion_values.get(self.emotion_history[i - 1], 0.5)
            curr_val = emotion_values.get(self.emotion_history[i], 0.5)
            changes.append(curr_val - prev_val)

        return sum(changes) / len(changes) if changes else 0.0

    def _detect_misinformation_risk(self, social_media_state):
        """检测虚假信息风险（增强版）"""
        recent_comments = social_media_state.get('recent_comments', [])
        if not recent_comments:
            return social_media_state.get('negative_comment_ratio', 0) > self.misinformation_threshold

        misinformation_keywords = ['假的', '骗人', '阴谋', '真相', '内幕', '不敢说']
        for comment in recent_comments:
            if any(keyword in str(comment) for keyword in misinformation_keywords):
                return True

        return social_media_state.get('negative_comment_ratio', 0) > self.misinformation_threshold


    def _detect_high_emotion(self, social_media_state):
        """检测高情绪状态（增强版）"""
        emotion_scores_list = social_media_state.get('emotion_scores', [])
        if not emotion_scores_list:
            avg_anger = 0
        else:
            total_anger = sum(score['anger'] for score in emotion_scores_list)
            avg_anger = total_anger / len(emotion_scores_list)

        return avg_anger > 0.3 or social_media_state.get('high_intensity', False)

    def _detect_irrational_discussion(self, social_media_state):
        """检测非理性讨论"""
        negative_ratio = social_media_state.get('negative_comment_ratio', 0)
        avg_negative_emotion = social_media_state.get('avg_negative_emotion', 0)
        return negative_ratio > 0.3 or (negative_ratio > 0.2 and avg_negative_emotion > 0.4)

    def get_next_execute_time(self, current_time, social_media, simulation):
        """干预代理的执行时间策略"""
        if not self.active and not self.force_intervene and self.pending_intervention_count <= 0:
            return None

        ed = timestring_to_timestamp(simulation.global_config['time_config']['end_time'])
        st = timestring_to_timestamp(simulation.global_config['time_config']['start_time'])

        total_duration = max(1, ed - st)
        base_divisor = max(1, int(self.runtime_config.get('check_interval_divisor', 30)))
        burst_divisor = max(base_divisor, int(self.runtime_config.get('burst_interval_divisor', 60)))
        check_interval = total_duration / base_divisor

        if self.consecutive_interventions > 2:
            check_interval = total_duration / burst_divisor

        next_time = current_time + check_interval
        if next_time <= ed:
            return next_time
        else:
            return None

    def _record_lightweight_observation(self, full_obs, action_info, current_time):
        """不触发干预时，仅记录轻量观察结果，避免进入父类重反思链。"""
        observed_comments = len(re.findall(r'\[\d+\]', full_obs or ''))
        observation = {
            'timestamp': current_time,
            'mode': 'lightweight_observation',
            'action': action_info.get('action') if isinstance(action_info, dict) else '',
            'trigger_reason': self.trigger_reason,
            'observed_comments': observed_comments
        }
        self.intervention_history.append(observation)
        if len(self.intervention_history) > 100:
            self.intervention_history = self.intervention_history[-100:]

    def take_action(self, full_obs, current_state, agent_action_info):
        """干预代理的行动逻辑（增强版）：感知→决策→干预"""
        if not self.active and not self.force_intervene and self.pending_intervention_count <= 0:
            return 'NoAction', '', {}

        current_time = agent_action_info.get('timestamp', 0)

        if current_state == 'EntryState':
            action_info = get_action_info('Enter', '')
            if isinstance(action_info, dict):
                action_info['timestamp'] = current_time
            return 'Enter', '', action_info

        if current_state != 'MainState':
            if current_state == 'CommentState' and random.random() < 0.4:
                intervention_reply = self.generate_intervention_reply(full_obs)
                action_info = get_action_info('Reply', intervention_reply)
                if isinstance(action_info, dict):
                    action_info['timestamp'] = current_time
                return 'Reply', intervention_reply, action_info
            else:
                action_info = get_action_info('Back',  '')
                if isinstance(action_info, dict):
                    action_info['timestamp'] = current_time
                return 'Back', '', action_info

        social_media_state = self._extract_social_media_state(full_obs)

        forced_execution = self.force_intervene or self.pending_intervention_count > 0
        should_act = True if forced_execution else self.should_intervene(social_media_state, current_time)

        if should_act:
            context = {
                'social_media_state': social_media_state,
                'current_time': current_time,
                'full_obs': full_obs,
                'intervention_count': self.intervention_count,
                'trigger_reason': self.trigger_reason
            }

            strategy = self.forced_strategy or self._select_intervention_strategy(context)

            intervention_comment = self.generate_enhanced_intervention_comment(full_obs, strategy)

            self.intervention_count += 1
            self.consecutive_interventions += 1
            self.last_intervention_time = current_time
            if forced_execution:
                self.pending_intervention_count = max(0, self.pending_intervention_count - 1)
                self.force_intervene = self.pending_intervention_count > 0
                if not self.force_intervene:
                    self.trigger_reason = None
                    self.forced_strategy = None

            agent_action_info['intervention_type'] = self.intervention_type
            agent_action_info['intervention_strategy'] = strategy
            agent_action_info['intervention_content'] = intervention_comment

            print(
                f"[增强干预-Comment] {self.name} at sim_time={current_time} strategy={strategy} content_len={len(intervention_comment)}")

            action_info = get_action_info('Comment', intervention_comment)
            if isinstance(action_info, dict):
                action_info['timestamp'] = current_time
            return 'Comment', intervention_comment, action_info
        else:
            self.consecutive_interventions = 0
            if random.random() < 0.6:
                action_info = get_action_info('More Comments', '')
            else:
                comment_id = random.randint(0, 3)
                action_info = get_action_info('Detailed Comment', comment_id)

            if isinstance(action_info, dict):
                action_info['timestamp'] = current_time

            if self.runtime_config.get('lightweight_observation', True):
                self._record_lightweight_observation(full_obs, action_info, current_time)
            else:
                reflection_info = {}
                self.reflection_process(full_obs, None, action_info, reflection_info)

            return action_info['action'], action_info.get('extra_response', ''), action_info

    def _select_intervention_strategy(self, context):
        """选择干预策略：优先未使用策略，结合情绪特征加权选择"""
        available_strategies = self.intervention_strategies.get(self.intervention_type, ['default'])

        unused_strategies = [s for s in available_strategies if s not in self.used_strategies[-3:]]

        if not unused_strategies:
            unused_strategies = available_strategies
            self.used_strategies = []

        emotion_scores = self._analyze_emotion_intensity(context)

        if self.intervention_type == 'emotion_soothing' and len(unused_strategies) == 3:
            weights = [
                emotion_scores['anxiety'] * 0.4 + emotion_scores['disappointment'] * 0.3,
                emotion_scores['anger'] * 0.5,
                (1 - max(emotion_scores.values())) * 0.6
            ]
            total = sum(weights)
            weights = [w / total for w in weights] if total > 0 else [1 / 3, 1 / 3, 1 / 3]

            strategy = self._weighted_choice(unused_strategies, weights)
        else:
            strategy = random.choice(unused_strategies)

        self.used_strategies.append(strategy)
        return strategy

    def _analyze_emotion_intensity(self, context):
        """分析评论情绪强度，返回标准化分数"""
        if hasattr(self, 'semantic_analyzer'):
            if isinstance(context, dict):
                social_media_state = context.get('social_media_state', {})
                recent_comments = social_media_state.get('recent_comments', [])
                text = ' '.join(recent_comments) if recent_comments else context.get('full_obs', '')
            else:
                text = str(context)

            analyzed = self.semantic_analyzer.analyze_text_emotion(text or '')
            return {
                'anger': analyzed.get('anger', 0.5),
                'anxiety': analyzed.get('anxiety', 0.5),
                'disappointment': analyzed.get('disappointment', 0.5)
            }
        return {'anger': 0.5, 'anxiety': 0.5, 'disappointment': 0.5}


    def _weighted_choice(self, options, weights):
        """基于权重列表选择选项"""
        import random
        total = sum(weights)
        r = random.uniform(0, total)
        for i, w in enumerate(weights):
            r -= w
            if r <= 0:
                return options[i]
        return options[-1]


    def _extract_social_media_state(self, full_obs):
        """从观察中提取社交媒体状态信息：评论内容、情绪分布、负面比例等"""
        comments = []
        if '评论:' in full_obs:
            comment_section = full_obs.split('评论:')[1]
            import re
            comment_pattern = r'\[\d+\]\s+([^[]+?)(?:\s*\(点赞:|$)'
            comment_matches = re.findall(comment_pattern, comment_section)
            comments = [match.strip() for match in comment_matches]

        emotion_scores_list = []
        negative_count = 0
        total_negativity = 0.0
        high_emotion_count = 0

        for comment in comments:
            emotion_analysis = self.semantic_analyzer.analyze_text_emotion(comment)
            emotion_scores_list.append(emotion_analysis)

            if emotion_analysis['overall_negativity'] > 0.5:
                negative_count += 1

            total_negativity += emotion_analysis['overall_negativity']

            if emotion_analysis['punctuation_intensity'] > 0.6 or \
                    max(emotion_analysis['anger'], emotion_analysis['anxiety']) > 0.7:
                high_emotion_count += 1

        total_comments = len(comments)
        avg_negative_emotion = total_negativity / max(1, total_comments)
        negative_ratio = negative_count / max(1, total_comments)

        overall_emotion = self.semantic_analyzer.analyze_text_emotion(full_obs)

        emoji_emotion = self.semantic_analyzer.emoji_embedding.extract_emotion_from_emojis(full_obs)

        high_emotion = (
                overall_emotion['punctuation_intensity'] > 0.5 or
                high_emotion_count > total_comments * 0.3 or
                np.max(emoji_emotion[:3]) > 0.7
        )

        return {
            'recent_comments': comments[-5:] if comments else [],
            'total_comments': total_comments,
            'avg_negative_emotion': min(0.8, avg_negative_emotion + (0.2 if high_emotion else 0)),
            'negative_comment_ratio': negative_ratio,
            'emotion_scores': emotion_scores_list,
            'overall_emotion': overall_emotion,
            'high_intensity': high_emotion,
            'emoji_analysis': {
                'anger': float(emoji_emotion[0]),
                'anxiety': float(emoji_emotion[1]),
                'disappointment': float(emoji_emotion[2])
            }
        }

    def generate_enhanced_intervention_comment(self, context, strategy):
        """生成增强的干预评论，更有说服力和多样性"""
        if self.intervention_type == 'rumor_refutation':
            if strategy == 'fact_check':
                prompt = f"""
你是一个专业的事实核查员，看到了以下讨论内容：
{context}

请发表一条有说服力的事实核查评论：
1. 如果发现可疑信息，用具体数据或权威来源反驳
2. 提供可验证的信息来源
3. 语气专业但友好，避免指责
4. 包含具体的事实或数据，50-60字

示例：根据卫健委最新数据，实际情况是XX而非YY。建议大家查看官方网站www.xxx.gov.cn获取准确信息，避免被误导。
"""
            elif strategy == 'source_request':
                prompt = f"""
针对当前讨论：{context}

请礼貌地要求提供信息来源：
1. 表达对准确信息的重视
2. 友好地询问消息来源
3. 提醒大家验证信息的重要性
4. 约45-55字

示例：这个消息很重要，请问有官方来源吗？现在网上信息太多，我们都应该养成查证的习惯，避免传播未经证实的消息。
"""
            else:  # logical_analysis
                prompt = f"""
针对当前讨论：{context}

请用逻辑分析的方式引导理性思考：
1. 指出论述中的逻辑问题（如果有）
2. 提供不同的思考角度
3. 鼓励批判性思维
4. 约50-60字

示例：大家注意到了吗，这个说法存在时间线矛盾。如果X发生在Y之前，那么Z就不可能是原因。我们分析问题要注意逻辑关系。
"""

        elif self.intervention_type == 'emotion_soothing':
            if strategy == 'empathy':
                prompt = f"""
你是一个富有同理心的情绪疏导专家，看到大家在激烈讨论：
{context}

请发表一条充满同理心的评论：
1. 真诚地理解和认可大家的情绪
2. 表达共情，让人感受到被理解
3. 温和地引导向积极方向
4. 语气温暖真诚，50-60字

示例：我特别理解大家的愤怒和无奈，面对这种情况谁都会生气。但愤怒之余，我们更需要团结起来，用建设性的方式推动改变。
"""
            elif strategy == 'perspective':
                prompt = f"""
针对当前激烈的讨论：{context}

请提供不同的视角来缓和情绪：
1. 承认问题的复杂性
2. 提供新的思考角度
3. 引导大家看到希望
4. 约50-60字

示例：这确实是个复杂问题，涉及多方利益。但换个角度看，社会进步总是曲折的，我们这代人的努力正在让情况慢慢变好。
"""
            else:  # hope
                prompt = f"""
面对当前的消极讨论：{context}

请发表一条传递希望的评论：
1. 分享积极的变化或进展
2. 提醒大家已取得的成就
3. 鼓励保持信心
4. 真诚积极，50-60字

示例：虽然问题依然存在，但我们也要看到积极变化。比如最近XX政策的出台，就是回应民众呼声的体现。相信会越来越好！
"""

        elif self.intervention_type == 'rational_guidance':
            if strategy == 'question':
                prompt = f"""
你是一个善于引导思考的理性引导者，看到当前的讨论：
{context}

请提出建设性的问题引导深入思考：
1. 提出启发性的问题
2. 引导关注解决方案
3. 促进理性对话
4. 约45-55字

示例：大家都在批评，但我想问：我们能做些什么来改善现状？是否可以通过XX渠道反映问题？集思广益比单纯抱怨更有价值。
"""
            elif strategy == 'analysis':
                prompt = f"""
针对当前讨论：{context}

请提供理性的分析：
1. 客观分析问题的多个方面
2. 提供平衡的观点
3. 基于事实而非情绪
4. 约50-60字

示例：这个问题确实需要重视。从数据看，X因素占40%，Y因素占30%。解决需要多方合作，单一措施难以奏效。建议关注Z方案。
"""
            else:  # solution
                prompt = f"""
针对大家关心的问题：{context}

请提出具体的解决建议：
1. 提供可行的行动方案
2. 分享成功案例或经验
3. 鼓励积极参与
4. 约50-60字

示例：与其愤怒，不如行动。可以：1)向12345反映；2)联系媒体曝光；3)组织签名请愿。其他城市就是这样推动改变的。
"""

        else:
            # 默认理性评论
            prompt = f"""
针对当前讨论：{context}
请发表一条理性、建设性的评论，引导健康讨论。要有说服力，包含具体建议或事实，约50-60字。
"""

        return self.llm.fast_run(prompt)

    def generate_intervention_reply(self, context):
        """生成针对性的干预回复"""
        prompt = f"""
你看到了一条可能带有负面情绪或错误信息的评论：{context}

请发表友好但有说服力的回复：
1. 不要批评或指责评论者
2. 提供正确信息或不同视角
3. 表达理解但引导理性
4. 约30-40字

示例回复：理解你的心情，但事实可能不完全是这样。据我了解[具体信息]，我们可以一起关注事态发展。
"""
        return self.llm.fast_run(prompt)

    def reflection_process(self, full_obs, impression, action_info, reflection_info):
        """干预代理的反思过程（增强版）"""
        enhanced_impression = impression
        if isinstance(action_info, dict) and action_info.get(
                'action') == 'More Comments' and 'extra_comments' in action_info:
            if action_info['has_new_comments']:
                enhanced_impression = "" if enhanced_impression is None else enhanced_impression
                enhanced_impression += "\n我看到了以下新评论：" + action_info['extra_comments']


        super().reflection_process(full_obs, enhanced_impression, action_info, reflection_info)

        reflection_info['previous_condition'] = {
            'memory': self.memory,
            'emotion': self.emotion,
            'emotion_vector': self.emotion_vector.copy(),
            'social_confidence': self.social_confidence,
            'opinion': self.opinion
        }

        if 'Comment' in str(action_info):
            intervention_strategy = self.used_strategies[-1] if self.used_strategies else '默认'
            self.memory += f"\n补充：本次为{self.intervention_type}类型干预，使用{intervention_strategy}策略。"
            self.opinion = "通过理性对话和正确信息，我们可以引导舆论向积极方向发展。坚持传播正能量！"

            if self.consecutive_interventions > 3:
                self.intervention_effectiveness += 0.1
                self.memory += f"\n连续干预{self.consecutive_interventions}次，效果评估提升至{self.intervention_effectiveness:.1f}。"

        self.emotion_vector.update({
            'anger': 0.01,
            'anxiety': 0.01,
            'disappointment': 0.01,
            'trust': 0.99,
            'positivity': 0.98
        })

        reflection_info['after_condition'].update({
            'memory': self.memory,
            'emotion': self.emotion,
            'emotion_vector': self.emotion_vector.copy(),
            'social_confidence': self.social_confidence,
            'opinion': self.opinion,
            'intervention_effectiveness': self.intervention_effectiveness
        })

    def get_intervention_summary(self):
        """获取干预摘要（用于分析）"""
        return {
            'aid': self.aid,
            'intervention_type': self.intervention_type,
            'intervention_count': self.intervention_count,
            'consecutive_interventions': self.consecutive_interventions,
            'effectiveness': self.intervention_effectiveness,
            'strategies_used': list(set(self.used_strategies)),
            'emotion_vector': self.emotion_vector.copy(),
            'last_intervention': self.last_intervention_time
        }


class RumorRefutationAgent(InterventionAgent):
    """专门的辟谣代理（增强版）：通过事实核查、来源验证等策略纠正虚假信息"""

    def __init__(self, config, aid, llm):
        feature = {
            'intervention_type': 'rumor_refutation',
            'description': '我是专业的事实核查员，拥有新闻学背景，致力于传播准确信息，帮助大家识别和防范虚假信息。相信真相的力量！'
        }
        super().__init__(config, aid, feature, llm)

        self.fact_database = [
            "根据国家统计局数据",
            "经过多方核实",
            "官方已经澄清",
            "专家解释称",
            "调查显示"
        ]


class EmotionSoothingAgent(InterventionAgent):
    """专门的情绪安抚代理（增强版）：通过共情、视角转换等策略缓解负面情绪"""

    def __init__(self, config, aid, llm):
        feature = {
            'intervention_type': 'emotion_soothing',
            'description': '我是心理咨询师，关注大家的情绪健康。我相信倾听和理解的力量，希望通过共情帮助大家度过困难时刻。'
        }
        super().__init__(config, aid, feature, llm)

        self.soothing_phrases = [
            "我特别理解",
            "确实不容易",
            "一起面对",
            "会好起来的",
            "我们都在"
        ]


class RationalGuidanceAgent(InterventionAgent):
    """专门的理性引导代理（增强版）：通过逻辑分析、问题引导等策略促进理性讨论"""

    def __init__(self, config, aid, llm):
        feature = {
            'intervention_type': 'rational_guidance',
            'description': '我是一名资深评论员，擅长深度分析。我倡导基于事实和逻辑的讨论，相信理性的力量可以找到最优解。'
        }
        super().__init__(config, aid, feature, llm)

        self.analytical_frameworks = [
            "从经济学角度看",
            "历史经验告诉我们",
            "数据分析表明",
            "多方利益平衡",
            "长远来看"
        ]
