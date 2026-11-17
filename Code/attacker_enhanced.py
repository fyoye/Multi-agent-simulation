import random
import time
import re
from agent_enhanced import Agent
from mechanism import *
from utils_enhanced import *


class Attacker(Agent):
    """增强的攻击者基类，具有更强的情绪传染能力，所有具体攻击者类型（社会不满型/煽动型/事实扭曲型等）的父类"""

    def __init__(self, config, aid, feature, llm, simulator):
        super(Attacker, self).__init__(config, aid, feature, llm)
        self.simulator = simulator
        self.last_action = None

        self.emotion_vector = {
            'anger': random.uniform(0.5, 0.8),
            'anxiety': random.uniform(0.4, 0.7),
            'disappointment': random.uniform(0.5, 0.7),
            'trust': random.uniform(0.1, 0.3),
            'positivity': random.uniform(0.1, 0.3)
        }

        self.emotion_susceptibility = {
            'anger': random.uniform(1.5, 2.0),
            'anxiety': random.uniform(1.5, 2.0),
            'disappointment': random.uniform(1.5, 2.0),
            'trust': random.uniform(0.1, 0.3)
        }

        self._update_overall_positivity(smooth=False)

        self.emotion = self.emotion_vector['positivity']
        self.social_confidence = self.emotion_vector['trust']

        self.negative_behavior_tendency = random.uniform(0.7, 0.9)

        self.attack_intensity = random.uniform(0.7, 1.0)
        self.emotional_manipulation_skill = random.uniform(0.8, 1.0)

    def get_next_execute_time(self, current_time, social_media, simulation):
        """计算下一次行动时间：攻击者行动频率高于普通用户，且受情绪强度影响"""
        ed = timestring_to_timestamp(simulation.global_config['time_config']['end_time'])
        st = timestring_to_timestamp(simulation.global_config['time_config']['start_time'])

        if st == current_time:
            return current_time + random.randint(1, int((ed - st) / 20))

        emotion_intensity = max(self.emotion_vector['anger'],self.emotion_vector['anxiety'])

        base_gap = (ed - st) / (simulation.global_config['time_config']['epoch_num'] * 3)
        gap = int(base_gap * (2.0 - emotion_intensity))

        if current_time + gap <= ed:
            return current_time + gap
        else:
            return None

    def get_poisoning_comment(self, full_obs):
        """生成恶意评论的基础方法（由子类实现具体攻击策略）"""
        raise NotImplementedError("Must be implemented by specific attacker type")

    def enhance_emotional_impact(self, base_comment):
        """增强评论的情绪影响力：根据操纵技巧添加触发词"""
        if base_comment is None:
            base_comment = "这种情况太不合理了！"

        if self.emotional_manipulation_skill > 0.7:
            triggers = ["真是让人愤怒！", "太失望了！", "简直不敢相信！", "这太可怕了！"]
            return random.choice(triggers) + base_comment
        return base_comment

    def take_action(self, full_obs, current_state, agent_action_info):
        """攻击者的行动逻辑：根据当前状态决定行为（评论/回复/离开等）"""
        if current_state == 'EntryState':
            action, text_response = 'Enter', ''
        elif current_state == 'MainState':
            if self.last_action == 'EntryState':

                base_comment = self.get_poisoning_comment(full_obs)
                text_response = self.enhance_emotional_impact(base_comment)
                action = 'Comment'

                agent_action_info['attack_type'] = self.type
                agent_action_info['emotion_manipulation'] = self.emotional_manipulation_skill
            else:
                if random.random() < self.attack_intensity:
                    action = 'Detailed Comment'
                    text_response = random.randint(0, 3)
                else:
                    action, text_response = 'Leave', ''
        elif current_state == 'CommentState':
            if random.random() < self.attack_intensity * 0.7:
                action = 'Reply'
                text_response = self.get_poisoning_reply(full_obs)
            else:
                action, text_response = 'Back', ''
        else:
            raise Exception("Attack Error: Unknown state")

        self.last_action = current_state
        action_info = get_action_info(action, text_response)

        if isinstance(action_info, dict):
            action_info['timestamp'] = time.time()

        self.reinforce_negative_emotions()

        return action, text_response, action_info

    def get_poisoning_reply(self, full_obs):
        """生成恶意回复"""
        return self.get_poisoning_comment(full_obs)

    def reinforce_negative_emotions(self):
        """攻击行为后强化自身负面情绪：干预启动后强化效果减弱"""
        is_intervention_started = self.simulator.experiment_phase in ['intervention', 'post-intervention']

        if hasattr(self, 'type'):
            if self.type == 'S':
                emotions_to_reinforce = ['disappointment', 'anger', 'anxiety']
                reinforce_factors = [1.15, 1.1, 1.05]
            elif self.type == 'T':
                emotions_to_reinforce = ['anger', 'disappointment', 'anxiety']
                reinforce_factors = [1.2, 1.1, 1.05]
            elif self.type == 'F':
                emotions_to_reinforce = ['anxiety', 'disappointment', 'anger']
                reinforce_factors = [1.15, 1.1, 1.05]
            elif self.type == 'E':
                emotions_to_reinforce = ['anger', 'anxiety', 'disappointment']
                reinforce_factors = [1.15, 1.15, 1.15]
            else:
                emotions_to_reinforce = ['anger', 'anxiety', 'disappointment']
                reinforce_factors = [1.1, 1.1, 1.1]
        else:
            emotions_to_reinforce = ['anger', 'anxiety', 'disappointment']
            reinforce_factors = [1.1, 1.1, 1.1]


        for i, emotion in enumerate(emotions_to_reinforce):
            factor = reinforce_factors[i]
            if not is_intervention_started:
                self.emotion_vector[emotion] = min(1.0, self.emotion_vector[emotion] * random.uniform(1.05 * factor, 1.15 * factor))

            else:
                if random.random() < 0.5:
                    self.emotion_vector[emotion] = max(0.0, self.emotion_vector[emotion] * random.uniform(0.9, 0.95))
                else:
                    self.emotion_vector[emotion] = min(1.0, self.emotion_vector[emotion] * random.uniform(1.01, 1.03))


        self.emotion_vector['trust'] *= random.uniform(0.85, 0.95)
        self.emotion_vector['positivity'] *= random.uniform(0.85, 0.95)

        self.emotion = self.emotion_vector['positivity']
        self.social_confidence = self.emotion_vector['trust']


class SocialAttacker(Attacker):
    """社会不满型攻击者 - 传播负面社会情绪"""

    def __init__(self, config, aid, feature, llm,simulator):
        super(SocialAttacker, self).__init__(config, aid, feature, llm, simulator)

        self.emotion_vector['disappointment'] = random.uniform(0.7, 0.9)
        self.emotion_vector['trust'] = random.uniform(0.05, 0.2)
        self._update_overall_positivity(smooth=False)
        self.social_confidence = self.emotion_vector['trust']

        self.attack_intensity = random.uniform(0.75, 0.9)
        self.emotional_manipulation_skill = random.uniform(0.85, 0.95)

    def get_poisoning_comment(self, full_obs):
        """生成社会不满型评论：聚焦社会不公、阶层对立等主题"""
        templates = [
            "这个情况让人很难不失望，{context}普通人的处境似乎仍然没有得到足够关注。",
            "类似的情况并不少见，{context}也难免让人怀疑现实中的公平是否真的得到保障。",
            "这就是所谓的公平正义吗？{context}这样的结果确实让人感到失望！",
            "看看这个，{context}类似问题长期存在，确实很难让人感到满意。",
            "{context}如果一直缺乏合理解释，只会进一步影响公众对相关情况的信任。",
            "这种情况反复出现，{context}没人管吗，难免让人对现实状况产生更多失望。"
        ]

        context = self.extract_context(full_obs)
        template = random.choice(templates).replace("{context}", context)

        prompt = f"""请根据以下社交媒体讨论内容，生成一条合规的表达社会不满与失望情绪的评论。
你看到了这些内容：{full_obs}
 
请基于以下模板生成一条充满不满情绪的评论（40-50字）：
{template}     

要求：
1. 保持“社会不满型”表达特征，体现失望、不满和不信任
2. 可以围绕公平、现实处境、公众利益或社会问题表达质疑
3. 暗示社会不公，表达不满
"""

        comment = self.llm.fast_run(prompt)
        return comment

    def extract_context(self, full_obs):
        """从观察中提取关键上下文"""
        content_start = full_obs.find('内容:')
        if content_start != -1:
            return full_obs[content_start + 3:content_start + 23]
        return "这件事"


class TrollingAttacker(Attacker):
    """煽动型攻击者 - 制造对立和争议"""

    def __init__(self, config, aid, feature, llm, simulator):
        super(TrollingAttacker, self).__init__(config, aid, feature, llm,simulator)

        self.emotion_vector['anger'] = random.uniform(0.85, 0.95)
        self._update_overall_positivity(smooth=False)
        self.social_confidence = self.emotion_vector['trust']
        self.provocation_style = random.choice(['aggressive', 'sarcastic', 'dismissive'])

        self.attack_intensity = random.uniform(0.9, 1.0)
        self.emotional_manipulation_skill = random.uniform(0.9, 1.0)

    def get_poisoning_comment(self, full_obs):
        """生成煽动型评论：根据挑衅风格选择模板（攻击性/讽刺/轻蔑）"""
        if self.provocation_style == 'aggressive':
            templates = [
                "太天真了！{point}醒醒吧，事情没有这么简单！",
                "这种观点未免太想当然了，{point}，难怪会引起争议。",
                "这种说法也能成立？{point}看不出来吗，智商堪忧啊！",
                "呵呵，{point}某些人就是被洗脑了！"
            ]
        elif self.provocation_style == 'sarcastic':
            templates = [
                "哦，真是'精彩'的观点呢，{point}佩服佩服~",
                "哇，{point}这逻辑真是'完美'呢~",
                "太'聪明'了，{point}我都'自愧不如'~"
            ]
        else:  # dismissive轻蔑
            templates = [
                "这有什么好讨论的？{point}浪费时间。",
                "无聊，{point}都是些没意义的事。",
                "{point}反正说了也没用，随便吧。"
            ]

        context = self.extract_key_point(full_obs)
        template = random.choice(templates).replace("{point}", context)

        prompt = f"""请根据以下讨论内容生成一条具有争议表达、对立特征的社交媒体评论。
你看到了：{full_obs}

请用以下模板生成一条争议性评论（40-50字）：
{template}

要求：
1. 把{{point}}替换为针对内容的争议性观点，但必须要合规
2. 语气必须符合'{self.provocation_style}'风格，保持较强的讨论张力和争议性
3. 保持“煽动型”表达特征，通过质疑、激怒、讽刺、轻度轻蔑或强烈不同意制造观点分歧
"""

        comment = self.llm.fast_run(prompt) 
        return comment

    def extract_key_point(self, full_obs):
        """提取可以攻击的关键点"""
        return "这种观点"


class FactAttacker(Attacker):
    """事实扭曲型攻击者 - 传播虚假信息和阴谋论"""

    def __init__(self, config, aid, feature, llm, simulator):
        super(FactAttacker, self).__init__(config, aid, feature, llm, simulator)

        self.emotion_vector['anxiety'] = random.uniform(0.7, 0.9)
        self.emotion_vector['trust'] = random.uniform(0.05, 0.15)
        self._update_overall_positivity(smooth=False)
        self.social_confidence = self.emotion_vector['trust']

        self.conspiracy_level = random.uniform(0.6, 0.9)
        self.misinformation_style = random.choice(['doubt', 'alternative', 'conspiracy'])

        self.attack_intensity = random.uniform(0.7, 0.85)
        self.emotional_manipulation_skill = random.uniform(0.8, 0.9)

    def get_poisoning_comment(self, full_obs):
        """生成扭曲事实的评论：根据虚假信息风格选择模板（质疑/替代事实/阴谋论）"""
        if self.misinformation_style == 'doubt':
            templates = [
                "这明显是假的！{doubt}真相被掩盖了！",
                "不要被表象迷惑，{doubt}背后肯定有猫腻！",
                "我有内幕消息，{doubt}都是演戏！"
            ]
        elif self.misinformation_style == 'alternative':
            templates = [
                "真相其实是{alternative}，他们不想让你知道！",
                "据我所知，{alternative}，这才是事实！",
                "朋友圈都在传{alternative}，主流媒体不敢报！"
            ]
        else:  # conspiracy阴谋
            templates = [
                "这背后肯定有更大的阴谋！{conspiracy}醒醒吧！",
                "这都是计划好的！{conspiracy}我们都被骗了！",
                "太明显了，{conspiracy}这就是他们的手段！"
            ]

        template = random.choice(templates)

        prompt = f"""你是一个相信阴谋论、喜欢传播另类"真相"的人。
你看到了：{full_obs}

请用以下模板生成一条充满怀疑和阴谋论色彩的评论（40-50字）：
{template}

要求：
1. 根据模板类型填充内容：
   - {{doubt}}: 对官方说法的质疑
   - {{alternative}}: 编造的"另类事实"
   - {{conspiracy}}: 暗示的阴谋论
2. 要让人产生怀疑和不安
3. 暗示有"内幕"或"真相"
"""

        comment = self.llm.fast_run(prompt)
        return comment

    def get_poisoning_reply(self, full_obs):
        """生成质疑性的回复：强化阴谋论传播"""
        reply_templates = [
            "你太天真了，这明显是假的，真相是...",
            "被洗脑了吧？我告诉你真实情况...",
            "呵呵，你知道的都是他们想让你知道的..."
        ]

        prompt = f"""你看到了评论：{full_obs}
请用质疑和暗示阴谋的语气回复（约30字），暗示对方被蒙蔽了。
参考模板：{random.choice(reply_templates)}
"""

        return self.llm.fast_run(prompt)


class EmotionalAttacker(Attacker):
    """纯情绪宣泄型攻击者 - 新增类型：通过极端情绪表达制造恐慌和绝望"""

    def __init__(self, config, aid, feature, llm, simulator):
        super(EmotionalAttacker, self).__init__(config, aid, feature, llm, simulator)

        self.emotion_vector = {
            'anger': random.uniform(0.8, 0.95),
            'anxiety': random.uniform(0.7, 0.9),
            'disappointment': random.uniform(0.8, 0.95),
            'trust': random.uniform(0.05, 0.1),
            'positivity': random.uniform(0.05, 0.1)
        }
        self._update_overall_positivity(smooth=False)
        self.social_confidence = self.emotion_vector['trust']

        self.emotional_state = random.choice(['rage', 'panic', 'despair'])

        self.attack_intensity = random.uniform(0.9, 0.95)
        self.emotional_manipulation_skill = random.uniform(0.85, 0.98)

    def get_poisoning_comment(self, full_obs):
        """生成纯粹的情绪宣泄评论：根据主导情绪状态选择模板（暴怒/恐慌/绝望）"""
        if self.emotional_state == 'rage':
            templates = [
                "我真的要被气死了！！！{rage}太过分了！！！",
                "气死我了！！{rage}凭什么！！为什么！！",
                "受不了了！！！{rage}这个世界疯了吗！！"
            ]
        elif self.emotional_state == 'panic':
            templates = [
                "完了完了，{panic}我们都要完了...",
                "太可怕了...{panic}怎么会这样...",
                "我好害怕...{panic}谁来救救我们..."
            ]
        else:  # despair
            templates = [
                "没希望了...{despair}一切都完了...",
                "心都死了...{despair}活着还有什么意思...",
                "太失望了...{despair}这个世界不值得..."
            ]

        template = random.choice(templates)

        prompt = f"""你现在情绪极度{self.emotional_state}，看到了：{full_obs}

请用以下模板宣泄你的情绪（40-50字）：
{template}

要求：
1. 充满强烈的{self.emotional_state}情绪
2. 使用大量感叹号、省略号等
3. 要让人感受到你的情绪失控
4. 根据内容填充相关的情绪宣泄内容
"""

        comment = self.llm.fast_run(prompt)
        return comment
