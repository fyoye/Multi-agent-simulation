import json
import re
import numpy as np
from typing import Dict, List, Optional


class EmotionAnalyzer:
    """
    情绪分析器：负责分析内容情绪、更新代理情绪状态、处理情绪传染,核心功能：文本情绪识别、代理情绪更新、情绪传染链检测
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.emotion_dimensions = ['anger', 'anxiety', 'disappointment', 'trust', 'positivity']
        self.negative_emotions = ['anger', 'anxiety', 'disappointment']
        self._emotion_cache = {}
        self._emotion_cache_max_size = 5000
        
    def analyze_content_emotion(self, content: str) -> Dict[str, float]:
        """
        使用LLM分析文本内容的情绪特征
        
        Args:
            content: 要分析的文本内容
            
        Returns:
            包含各情绪维度得分的字典
        """
        cache_key = (content or '').strip()
        if cache_key in self._emotion_cache:
            return self._emotion_cache[cache_key].copy()


        prompt = f"""      
分析以下内容的情绪特征，返回JSON格式的情绪评分。

内容：{content}

请评估以下情绪维度（0-1范围，0表示没有，1表示非常强烈）：
- anger: 愤怒程度（攻击性、敌意、批判）
- anxiety: 焦虑程度（担忧、恐慌、不安）
- disappointment: 失望程度（沮丧、无奈、消极）
- positivity: 积极/建设性表达程度（理性、安抚、中性建议、积极支持）
- negativity: 整体负面程度
- misinformation: 虚假信息可能性（夸大、歪曲、谣言特征）

只返回JSON格式，不要其他解释：
{{"anger": 0.x, "anxiety": 0.x, "disappointment": 0.x, "positivity": 0.x, "negativity": 0.x, "misinformation": 0.x}}
"""
        
        try:
            result = self.llm.fast_run(prompt)
            json_match = re.search(r'\{[^}]+\}', result)
            if json_match:
                emotion_scores = json.loads(json_match.group())
                required_scores = {
                    'anger': 0.1,
                    'anxiety': 0.1,
                    'disappointment': 0.1,
                    'negativity': 0.1,
                    'misinformation': 0.0
                }
                for key, default in required_scores.items():
                    emotion_scores[key] = emotion_scores.get(key, default)

                if 'positivity' not in emotion_scores:
                    negative_average = (
                        float(emotion_scores.get('anger', 0.0)) +
                        float(emotion_scores.get('anxiety', 0.0)) +
                        float(emotion_scores.get('disappointment', 0.0)) +
                        float(emotion_scores.get('negativity', 0.0))
                    ) / 4
                    emotion_scores['positivity'] = 1.0 - negative_average

                for key in emotion_scores:
                    emotion_scores[key] = max(0.0, min(1.0, float(emotion_scores[key])))
                self._cache_emotion_scores(cache_key, emotion_scores)
                return emotion_scores
            else:
                return self._get_default_emotion_scores()
                
        except Exception as e:
            print(f"情绪分析失败: {e}")
            return self._get_default_emotion_scores()

    def _cache_emotion_scores(self, cache_key: str, emotion_scores: Dict[str, float]):
        if not cache_key:
            return

        if len(self._emotion_cache) >= self._emotion_cache_max_size:
            oldest_key = next(iter(self._emotion_cache))
            self._emotion_cache.pop(oldest_key, None)

        self._emotion_cache[cache_key] = emotion_scores.copy()
    
    def _get_default_emotion_scores(self) -> Dict[str, float]:
        """返回默认的中性情绪分数"""
        return {
            'anger': 0.1,
            'anxiety': 0.1,
            'disappointment': 0.1,
            'positivity': 0.5,
            'negativity': 0.1,
            'misinformation': 0.0
        }
    
    def update_agent_emotion(self, agent, perceived_emotions: Dict[str, float], 
                           own_action_emotion: Optional[Dict[str, float]] = None,is_intervention_started: bool = False):  # 新增参数：干预是否已启动
        """
        根据感知内容和自身行为更新代理的情绪状态
        核心逻辑：负面情绪传染（感知情绪→个体易感性→情绪共振）→ 信任度调整→行为自我强化→整体积极性更新→情绪衰减
        Args:
            agent: 要更新情绪的代理
            perceived_emotions: 从感知内容中分析出的情绪
            own_action_emotion: 代理自身行为表达的情绪（可选）
        """
        for emotion in self.negative_emotions:
            if emotion in perceived_emotions:

                base_contagion = perceived_emotions[emotion]

                susceptibility = agent.emotion_susceptibility.get(emotion, 1.0)

                current_level = agent.emotion_vector[emotion]

                if not is_intervention_started:
                    resonance = 1.0 + 0.6 * current_level
                else:
                    resonance = 1.0 + 0.2 * current_level

                resistance_factor = 1.0 - getattr(agent, 'emotion_resistance', 0.0)

                delta = base_contagion * susceptibility * resonance * 0.2 * resistance_factor

                new_value = agent.emotion_vector[emotion] + delta
                agent.emotion_vector[emotion] = self._sigmoid_normalize(new_value)

        intervention_support = max(
            perceived_emotions.get('intervention_trust_boost', 0.0),
            perceived_emotions.get('intervention_positivity_boost', 0.0)
        )

        misinformation = perceived_emotions.get('misinformation', 0.0)
        negative_pressure = (
            perceived_emotions.get('anger', 0.0) * 0.4 +
            perceived_emotions.get('anxiety', 0.0) * 0.3 +
            perceived_emotions.get('disappointment', 0.0) * 0.3
        )
        trust_decay = misinformation * 0.18
        if negative_pressure > 0.45:
            trust_decay += (negative_pressure - 0.45) * 0.04
        if intervention_support > 0 and trust_decay > 0:
            mitigation = min(0.35, intervention_support * 0.25)
            trust_decay *= (1.0 - mitigation)
        if trust_decay > 0:
            agent.emotion_vector['trust'] = max(0.0, agent.emotion_vector['trust'] - trust_decay)

        if intervention_support > 0:
            trust_gain = min(0.07, intervention_support * 0.07)
            agent.emotion_vector['trust'] = min(1.0, agent.emotion_vector['trust'] + trust_gain)

        if own_action_emotion:
            for emotion, value in own_action_emotion.items():
                if emotion in agent.emotion_vector and value > 0.7:
                    # 强烈情绪表达会强化自身对应情绪
                    reinforcement = 1.1 + 0.1 * value
                    agent.emotion_vector[emotion] = min(1.0, 
                        agent.emotion_vector[emotion] * reinforcement)

        self._update_overall_positivity(agent)

        if is_intervention_started:
            self._apply_emotion_decay(agent, is_intervention_started)
    
    def _sigmoid_normalize(self, value: float, k: float = 5.0) -> float:
        """使用sigmoid函数将值归一化到[0,1]区间"""
        return 1.0 / (1.0 + np.exp(-k * (value - 0.5)))
    
    def _update_overall_positivity(self, agent):
        """根据各负面情绪更新整体积极性"""
        negative_weights = {'anger': 0.4, 'anxiety': 0.3, 'disappointment': 0.3}
        negative_sum = sum(
            agent.emotion_vector[emotion] * negative_weights.get(emotion, 0.33)
            for emotion in self.negative_emotions
        )
        
        target_positivity = 0.7 * (1.0 - negative_sum) + 0.3 * agent.emotion_vector['trust']
        current_positivity = agent.emotion_vector.get('positivity', 0.5)

        agent.emotion_vector['positivity'] = max(0.0, min(
            1.0,
            0.6 * current_positivity + 0.4 * target_positivity
        ))

        agent.emotion = agent.emotion_vector['positivity']
    
    def _apply_emotion_decay(self, agent, is_intervention_phase: bool = False):
        """应用情绪自然衰减，避免情绪无限累积"""
        decay_rate = 0.09 if is_intervention_phase else 0.02

        for emotion in self.negative_emotions:
            baseline = 0.1
            current = agent.emotion_vector[emotion]
            agent.emotion_vector[emotion] = current - decay_rate * (current - baseline)

        trust_baseline = 0.5
        current_trust = agent.emotion_vector['trust']
        if current_trust < trust_baseline:
            recovery_rate = 0.04 if is_intervention_phase else 0.02
            agent.emotion_vector['trust'] = min(trust_baseline, 
                current_trust + recovery_rate)
    
    def calculate_emotion_distance(self, agent1, agent2) -> float:
        """计算两个代理之间的情绪距离（用于情绪聚类分析）
        距离定义：各情绪维度差值平方和的平方根（欧氏距离），值越小情绪越相似"""
        distance = 0.0
        for emotion in self.emotion_dimensions:
            if emotion in agent1.emotion_vector and emotion in agent2.emotion_vector:
                diff = agent1.emotion_vector[emotion] - agent2.emotion_vector[emotion]
                distance += diff ** 2
        return np.sqrt(distance)
    
    def get_emotion_summary(self, agent: object) -> str:
        """获取代理当前情绪状态的文字描述"""
        emotions = agent.emotion_vector if hasattr(agent, 'emotion_vector') else {}

        dominant_emotion = None
        max_value = 0.3
        
        for emotion in self.negative_emotions:
            if emotions[emotion] > max_value:
                max_value = emotions[emotion]
                dominant_emotion = emotion

        if dominant_emotion == 'anger':
            intensity = "愤怒" if max_value > 0.7 else "不满"
            return f"情绪激动，充满{intensity}"
        elif dominant_emotion == 'anxiety':
            intensity = "极度焦虑" if max_value > 0.7 else "担忧"
            return f"感到{intensity}和不安"
        elif dominant_emotion == 'disappointment':
            intensity = "非常失望" if max_value > 0.7 else "有些沮丧"
            return f"{intensity}，情绪低落"
        elif emotions['trust'] < 0.3:
            return "对社会充满不信任"
        else:
            return "情绪相对平稳"
    
    def detect_emotional_contagion_chains(self, agents: List, 
                                        interaction_history: List) -> List[Dict]:
        """
        检测情绪传染链（识别情绪从源头代理传播到其他代理的路径）
        检测逻辑：分析交互历史，若代理B在与代理A交互后，同类负面情绪超过A的70%，则认为A是B的情绪传染源，形成传染链
        Returns:
            传染链列表，每个链包含源头、受影响代理和传染路径
        """
        chains = []

        for i, interaction in enumerate(interaction_history):
            if i == 0:
                continue
                
            source_agent = interaction.get('agent')
            if not source_agent:
                continue

            affected_agents = []
            for j in range(i + 1, min(i + 10, len(interaction_history))):
                target_interaction = interaction_history[j]
                target_agent = target_interaction.get('agent')

                if target_agent and self._is_emotionally_influenced(
                    source_agent, target_agent, interaction, target_interaction):
                    affected_agents.append({
                        'agent': target_agent,
                        'influence_strength': self._calculate_influence_strength(
                            source_agent, target_agent),
                        'time_gap': j - i
                    })
            
            if affected_agents:
                chains.append({
                    'source': source_agent,
                    'affected': affected_agents,
                    'dominant_emotion': self._get_dominant_emotion(source_agent),
                    'start_time': interaction.get('timestamp')
                })
        
        return chains
    
    def _is_emotionally_influenced(self, source, target, 
                                  source_interaction, target_interaction) -> bool:
        """判断目标代理是否受到源代理的情绪影响"""
        source_emotion = source_interaction.get('expressed_emotion', {})
        target_emotion = target_interaction.get('expressed_emotion', {})
        
        for emotion in self.negative_emotions:
            if (source_emotion.get(emotion, 0) > 0.5 and
                target_emotion.get(emotion, 0) > source_emotion.get(emotion, 0) * 0.7):
                return True
        return False
    
    def _calculate_influence_strength(self, source, target) -> float:
        """计算情绪影响强度"""
        similarity = 1.0 - self.calculate_emotion_distance(source, target)
        return max(0.0, similarity)
    
    def _get_dominant_emotion(self, agent) -> str:
        """获取代理的主导情绪"""
        max_emotion = 'neutral'
        max_value = 0.3
        
        for emotion in self.negative_emotions:
            if agent.emotion_vector[emotion] > max_value:
                max_value = agent.emotion_vector[emotion]
                max_emotion = emotion
        
        return max_emotion
