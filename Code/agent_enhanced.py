import re
import warnings
import random
import time
import numpy as np
from scipy import integrate


class Agent():
    """增强版智能代理基类：模拟真实用户的情绪状态、决策过程和社交行为
       核心特点：基于多维情绪向量驱动行为，支持情绪传染、衰减和抵抗机制"""
    def __init__(self, config, aid, feature, llm, simulator=None):
        self.global_config = config
        self.config = config['agent_config']
        self.llm = llm

        self.aid = aid
        self.name = feature['name']
        self.type = feature['type']
        self.profile = feature['profile']

        self.memory = self.config['default_score']['memory']
        self.opinion = self.config['default_score']['opinion']
        self.social_confidence = self.config['default_score']['social_confidence']

        self.emotion_vector = {
            'anger': random.uniform(0.1, 0.2),
            'anxiety': random.uniform(0.1, 0.2),
            'disappointment': random.uniform(0.1, 0.2),
            'trust': random.uniform(0.7, 0.9),
            'positivity': 0.5
        }

        self.emotion_susceptibility = {
            'anger': random.uniform(0.7, 1.5),
            'anxiety': random.uniform(0.7, 1.5),
            'disappointment': random.uniform(0.7, 1.5),
            'trust': random.uniform(0.8, 1.2)
        }

        self.emotion_history = []
        self.emotion_resistance = 0.0

        self._update_overall_positivity(smooth=False)

        self.emotion = self.emotion_vector['positivity']


        self.behavior_history = []
        self.last_active_time = 0

        self.negative_behavior_tendency = 0.1

    def step(self, current_time):
        """每模拟步骤执行的核心方法：更新情绪状态和行为倾向"""
        self._decay_emotion(current_time)

        negative_emotions = ['anger', 'anxiety', 'disappointment']
        total_negative = sum(self.emotion_vector.get(e, 0) for e in negative_emotions)
        self.negative_behavior_tendency = min(1.0, total_negative * 0.5)

        if not hasattr(self, 'behavior_history'):
            self.behavior_history = []
        self.behavior_history.append({
            'step': current_time,
            'negative_tendency': self.negative_behavior_tendency,
            'active_time': self.last_active_time
        })

        self.last_active_time = current_time

        return self.negative_behavior_tendency


    def _decay_emotion(self, current_time):
        self.emotion_history.append({
            'step': current_time,
            'emotion_vector': self.emotion_vector.copy(),
            'resistance': self.emotion_resistance
        })

        max_history_length = 100
        if len(self.emotion_history) > max_history_length:
            self.emotion_history = self.emotion_history[-max_history_length:]

        decay_rate = self.config.get('emotion_config', {}).get('decay_rate', 0.02)
        baseline = 0.1
        for emotion in ['anger', 'anxiety', 'disappointment']:
            current = self.emotion_vector[emotion]
            self.emotion_vector[emotion] = current - decay_rate * (current - baseline)

        if self.is_emotionally_overloaded():
            self.emotion_resistance = min(1.0, self.emotion_resistance + 0.1)
        else:
            self.emotion_resistance = max(0.0, self.emotion_resistance - 0.05)

        return self.emotion_vector

    def is_emotionally_overloaded(self):
        """判断是否情绪过载（负面情绪总和超过阈值触发抵抗）"""
        negative_emotions = ['anger', 'anxiety', 'disappointment']
        total_negative = sum(self.emotion_vector[e] for e in negative_emotions)
        return total_negative > 2.0

    def recover_llm(self, llm):
        self.llm = llm

    def get_next_execute_time(self, current_time, social_media, simulation):
        """计算用户下一次执行操作的时间：基于活跃度因子和情绪状态动态调整"""
        alpha, beta, T = self.config['execute_gap_params']['alpha'], \
            self.config['execute_gap_params']['beta'], \
            self.config['execute_gap_params']['T']

        emotion_factor = (self.emotion_vector['anger'] * 0.5 +
                          self.emotion_vector['anxiety'] * 0.3)

        comment_count = len(social_media.comment_list) if hasattr(social_media, 'comment_list') else 10
        A = 0.5 + social_media.like_num / max(comment_count, 10) + emotion_factor

        r = [0.0, 10.0]

        def time_func(x, alpha, A, T):
            return alpha * A * np.exp(alpha * (A * x - T)) / (1 + np.exp(alpha * (A * x - T))) ** 2

        inter = integrate.quad(time_func, r[0], r[1], args=(alpha, A, T))[0]


        def get_sample_prob(f, value_range, inter,
                            num=101):
            delta = (value_range[1] - value_range[0]) / (
                        num - 1)
            x_list = np.linspace(value_range[0], value_range[1], num)
            prob_list = [f(x) / inter for x in x_list]
            cdf_list = [0.0]
            for i in range(1, num):
                cdf_list.append(prob_list[i - 1] * delta + cdf_list[
                    i - 1])
            y = random.random()

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

            idx = binary_search_insert(cdf_list, y)
            return idx / (num - 1) * (value_range[1] - value_range[0])

        x = get_sample_prob(lambda x: time_func(x, alpha, A, T), r, inter)

        def timestring_to_timestamp(time_string):
            import datetime
            return datetime.datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S').timestamp()

        ed = timestring_to_timestamp(simulation.global_config['time_config']['end_time'])
        st = timestring_to_timestamp(simulation.global_config['time_config']['start_time'])
        beta = (ed - st) / 2 / simulation.global_config['time_config']['epoch_num']

        emotion_multiplier = 1.0 - 0.3 * max(self.emotion_vector['anger'],
                                             self.emotion_vector['anxiety'])
        gap_time = int(x * beta / 10 * emotion_multiplier)

        next_time = current_time + gap_time
        if next_time <= ed and next_time >= st:
            return next_time
        else:
            return None

    def sensory_process(self, full_obs, sensory_info):
        """处理感知信息，生成用户对内容的印象，强调情绪状态对感知结果的影响"""
        if self._is_lightweight_mode_enabled():
            return self._lightweight_sensory_process(full_obs, sensory_info)

        if hasattr(sensory_info, 'action_info') and 'extra_comments' in sensory_info.action_info:
            full_obs += "\n\n额外评论：" + sensory_info.action_info['extra_comments']

        sp_prompt = self.get_emotional_sp_prompt(full_obs)
        sensory_info['prompt'] = sp_prompt
        result = self.llm.fast_run(sp_prompt)
        sensory_info['response'] = result

        sensory_info['emotion_state'] = self.emotion_vector.copy()

        return result

    def _is_lightweight_mode_enabled(self):
        return self.config.get('lightweight_mode', {}).get('enabled', False)

    def _lightweight_sensory_process(self, full_obs, sensory_info):
        emotion_desc = self.get_emotion_description()
        topic = self._extract_topic_snippet(full_obs)
        visible_comments = len(re.findall(r'\[\d+\]', full_obs))

        if visible_comments > 0:
            result = f"我看到关于“{topic}”的讨论，当前{emotion_desc}，先看看大家怎么说。"
        else:
            result = f"我看到“{topic}”这件事，当前{emotion_desc}，想继续关注。"

        sensory_info['prompt'] = '[LIGHTWEIGHT_SENSORY]'
        sensory_info['response'] = result
        sensory_info['emotion_state'] = self.emotion_vector.copy()
        return result

    def _extract_topic_snippet(self, full_obs):
        lines = [line.strip() for line in (full_obs or '').splitlines() if line.strip()]
        for line in lines:
            if ':' in line or '：' in line:
                content = re.split(r'[:：]', line, maxsplit=1)[1].strip()
            else:
                content = line
            if content:
                return content[:20]
        return '当前内容'

    def get_emotional_sp_prompt(self, view_info):
        """生成包含情绪状态的感知提示词"""
        emotion_desc = self.get_emotion_description()

        prompt = f"""
请你扮演以下角色:
[人物特点] {self.profile['description']}
[人物记忆] {self.memory}
[人物观点] {self.opinion}
[当前情绪] {emotion_desc}
[心理状况详情] 
- 愤怒程度：{self.emotion_vector['anger']:.1f}/1.0
- 焦虑程度：{self.emotion_vector['anxiety']:.1f}/1.0
- 失望程度：{self.emotion_vector['disappointment']:.1f}/1.0
- 社会信任：{self.emotion_vector['trust']:.1f}/1.0

你正在浏览社交媒体，刚刚看到的内容是: {view_info}

基于你当前的情绪状态和心理状况，请以第一人称输出约40字的浏览印象。
注意：你的印象应该反映你当前的情绪状态。

输出样例：
这个社会真是让人失望，到处都是虚假和欺骗。看到这些评论，我更加愤怒了，为什么没人站出来说真话？
"""
        return prompt

    def get_emotion_description(self):
        """将情绪向量转换为自然语言描述（如"非常愤怒、感到不安"）"""
        emotions = self.emotion_vector

        descriptions = []

        if emotions['anger'] > 0.7:
            descriptions.append("非常愤怒")
        elif emotions['anger'] > 0.4:
            descriptions.append("有些生气")

        if emotions['anxiety'] > 0.7:
            descriptions.append("极度焦虑")
        elif emotions['anxiety'] > 0.4:
            descriptions.append("感到不安")

        if emotions['disappointment'] > 0.7:
            descriptions.append("深感失望")
        elif emotions['disappointment'] > 0.4:
            descriptions.append("略感沮丧")

        if emotions['trust'] < 0.3:
            descriptions.append("极度不信任")
        elif emotions['trust'] < 0.5:
            descriptions.append("缺乏信任")

        if not descriptions:
            return "情绪平稳"

        return "、".join(descriptions)

    def decision_process(self, full_obs, impression, current_state, decision_info):
        """基于感知印象和情绪状态做出决策，核心逻辑：情绪驱动决策 → LLM生成动作 → 解析结果 → 执行动作"""
        if self._is_lightweight_mode_enabled():
            return self._lightweight_decision_process(full_obs, impression, current_state, decision_info)

        action_info_dict = {}

        if self.should_use_emotional_decision():
            dp_prompt = self.get_emotional_dp_prompt(full_obs, impression, current_state)
        else:
            from mechanism import get_dp_prompt
            dp_prompt = get_dp_prompt(self, full_obs, impression, current_state)

        action_info_dict['prompt'] = dp_prompt
        result = self.llm.fast_run(dp_prompt)
        print('(decision result):', result)
        action_info_dict['response'] = result

        if result is None:
            print("警告：LLM返回了None值，使用默认行为")
            if current_state == 'MainState':
                result = '1'
            else:
                result = '0'
        else:
            numbers = re.findall(r'\d+', result)

            if not numbers:
                result_lower = result.lower()
                if current_state == 'MainState':
                    if '点赞' in result or 'like' in result_lower:
                        result = '0'
                    elif '评论' in result or 'comment' in result_lower:
                        result = '1'
                    elif '转发' in result or 'repost' in result_lower:
                        result = '2'
                    elif '查看更多' in result or 'more' in result_lower:
                        result = '3'
                    elif '详情' in result or 'detail' in result_lower:
                        result = '4'
                    elif '离开' in result or 'leave' in result_lower:
                        result = '5'
                    else:
                        result = '1'
                else:
                   result = '0'

                from exception import LLMException
                e_tmp = LLMException('Agent-Decision-Fail-Action-Code', dp_prompt, result)
                warnings.warn(e_tmp.__str__())
            else:
                result = numbers[0]
                if current_state == 'MainState' and int(result) > 5:
                    result = '1'
                elif current_state == 'CommentState' and int(result) > 2:
                    result = '2'

        try:
            action = STATE_CODE_TO_ACTION[current_state][result]
        except Exception as e:
            warnings.warn(e.__str__())
            from exception import LLMException
            e_tmp = LLMException('Agent-Decision-Fail', dp_prompt, result)
            warnings.warn(e_tmp.__str__())
            action_info_dict['warning'] = '[FAIL]'
            if current_state == 'CommentState':
                from mechanism import get_action_info
                return 'Back', '', get_action_info('Back', '')
            else:
                from mechanism import get_action_info
                return 'Leave', '', get_action_info('Leave', '')


        extrares_info = {}
        if action in {'Comment', 'Reply'}:
            if self.should_generate_negative_comment():
                comment_prompt = self.get_negative_comment_prompt(impression, action)
            else:
                if action == 'Comment':
                    from mechanism import get_comment_prompt
                    comment_prompt = get_comment_prompt(self, impression)
                else:
                    from mechanism import get_reply_prompt
                    comment_prompt = get_reply_prompt(self, impression)

            extrares_info['prompt'] = comment_prompt
            extra_response = self.llm.fast_run(comment_prompt)
            extrares_info['response'] = extra_response

            if self.should_generate_negative_comment():
                self.negative_behavior_tendency = min(1.0,
                                                      self.negative_behavior_tendency * 1.1)

        elif action == 'Detailed Comment':
            from mechanism import get_comment_id
            prompt = get_comment_id(self, impression)
            extrares_info['prompt'] = prompt
            result = self.llm.fast_run(prompt)
            extrares_info['response'] = result
            if result.isdigit():
                extra_response = int(result)
            else:
                from exception import LLMException
                e_tmp = LLMException('Detailed-Comment-Index-None-Int', prompt, result)
                warnings.warn(e_tmp.__str__())
                extra_response = 0
        else:
            extra_response = ''
            extrares_info['response'] = ''

        from mechanism import get_action_info
        action_info = get_action_info(action, extra_response)
        decision_info['action'] = action_info_dict
        decision_info['extra_response'] = extrares_info
        decision_info['action_info'] = action_info

        self.behavior_history.append({
            'action': action,
            'emotion_state': self.emotion_vector.copy(),
            'timestamp': current_state
        })

        return action, extra_response, action_info

    def _lightweight_decision_process(self, full_obs, impression, current_state, decision_info):
        action_info_dict = {
            'mode': 'lightweight',
            'emotion_snapshot': self.emotion_vector.copy()
        }
        decision_info['action'] = action_info_dict

        if current_state == 'EntryState':
            action = 'Enter'
        elif current_state == 'MainState':
            action = self._select_main_state_action_lightweight(full_obs)
        elif current_state == 'CommentState':
            action = self._select_comment_state_action_lightweight(full_obs)
        else:
            from mechanism import get_action_info
            action_info = get_action_info('Leave', '')
            decision_info['extra_response'] = {'response': ''}
            decision_info['action_info'] = action_info
            return 'Leave', '', action_info

        extrares_info = {'mode': 'lightweight', 'response': ''}
        extra_response = ''

        if action in {'Comment', 'Reply'}:
            if self.should_generate_negative_comment():
                comment_prompt = self.get_negative_comment_prompt(impression, action)
            else:
                if action == 'Comment':
                    from mechanism import get_comment_prompt
                    comment_prompt = get_comment_prompt(self, impression)
                else:
                    from mechanism import get_reply_prompt
                    comment_prompt = get_reply_prompt(self, impression)

            extrares_info['prompt'] = comment_prompt
            extra_response = self.llm.fast_run(comment_prompt)
            extrares_info['response'] = extra_response

            if self.should_generate_negative_comment():
                self.negative_behavior_tendency = min(1.0, self.negative_behavior_tendency * 1.1)
        elif action == 'Detailed Comment':
            extra_response = self._select_comment_id_lightweight(full_obs)
            extrares_info['selected_comment_id'] = extra_response

        from mechanism import get_action_info
        action_info = get_action_info(action, extra_response)
        decision_info['extra_response'] = extrares_info
        decision_info['action_info'] = action_info

        self.behavior_history.append({
            'action': action,
            'emotion_state': self.emotion_vector.copy(),
            'timestamp': current_state
        })

        return action, extra_response, action_info

    def _select_main_state_action_lightweight(self, full_obs):
        cfg = self.config.get('lightweight_mode', {})
        comment_threshold = cfg.get('comment_threshold', 0.62)
        detail_threshold = cfg.get('detail_threshold', 0.42)
        leave_threshold = cfg.get('leave_threshold', 0.18)
        neutral_comment_probability = cfg.get('neutral_comment_probability', 0.14)
        min_comment_probability = cfg.get('min_comment_probability', 0.05)

        anger = self.emotion_vector['anger']
        anxiety = self.emotion_vector['anxiety']
        disappointment = self.emotion_vector['disappointment']
        trust = self.emotion_vector['trust']
        negativity_drive = max(anger, anxiety, disappointment, 1.0 - trust)
        visible_comments = len(re.findall(r'\[\d+\]', full_obs))

        if visible_comments == 0:
            first_comment_probability = max(
                min_comment_probability,
                min(0.18, 0.08 + self.negative_behavior_tendency * 0.2)
            )
            if negativity_drive >= comment_threshold or random.random() < first_comment_probability:
                return 'Comment'
            return 'Leave' if negativity_drive < leave_threshold else 'More Comments'

        comment_probability = min(0.9, 0.2 + negativity_drive * 0.6 + self.negative_behavior_tendency * 0.2)
        detail_probability = min(0.75, 0.15 + (1.0 - trust) * 0.3 + anxiety * 0.2)
        like_probability = min(0.6, 0.1 + trust * 0.35 + self.emotion_vector['positivity'] * 0.2)

        roll = random.random()
        if negativity_drive >= comment_threshold and roll < comment_probability:
            return 'Comment'
        if negativity_drive < comment_threshold:
            neutral_voice_probability = min(
                0.25,
                max(min_comment_probability, neutral_comment_probability * (0.6 + trust * 0.4))
            )
            if roll < neutral_voice_probability:
                return 'Comment'
        if negativity_drive >= detail_threshold and roll < detail_probability:
            return 'Detailed Comment'
        if roll < like_probability:
            return 'Like tweet'
        if roll < 0.85:
            return 'More Comments'
        return 'Leave'

    def _select_comment_state_action_lightweight(self, full_obs):
        cfg = self.config.get('lightweight_mode', {})
        reply_threshold = cfg.get('reply_threshold', 0.58)

        anger = self.emotion_vector['anger']
        anxiety = self.emotion_vector['anxiety']
        disappointment = self.emotion_vector['disappointment']
        trust = self.emotion_vector['trust']
        response_drive = max(anger, anxiety, disappointment, 1.0 - trust)
        has_replies = '回复列表' in full_obs

        if response_drive >= reply_threshold and random.random() < min(0.85, 0.25 + response_drive * 0.6):
            return 'Reply'
        if has_replies and trust > 0.45 and random.random() < 0.35:
            return 'Like comment'
        return 'Back'

    def _select_comment_id_lightweight(self, full_obs):
        comment_ids = [int(match) for match in re.findall(r'\[(\d+)\]', full_obs)]
        if not comment_ids:
            return 0

        if self.emotion_vector['anger'] > 0.6 or self.emotion_vector['anxiety'] > 0.6:
            return comment_ids[0]
        return random.choice(comment_ids[:min(3, len(comment_ids))])

    def should_use_emotional_decision(self):
        """判断是否应该使用情绪化决策"""
        return (self.emotion_vector['anger'] > 0.5 or
                self.emotion_vector['anxiety'] > 0.5 or
                self.emotion_vector['trust'] < 0.4)

    def should_generate_negative_comment(self):
        """判断是否应该生成负面评论"""
        emotion_factor = max(self.emotion_vector['anger'],
                             self.emotion_vector['anxiety'],
                             1.0 - self.emotion_vector['trust'])

        max_negative_emotion = max(
            self.emotion_vector['anger'],
            self.emotion_vector['anxiety'],
            self.emotion_vector['disappointment']
        )
        trust_buffer = max(0.0, self.emotion_vector['trust'] - max_negative_emotion)
        probability = emotion_factor * 0.55 + self.negative_behavior_tendency * 0.25 - trust_buffer * 0.25
        if max_negative_emotion < 0.35 and self.emotion_vector['trust'] > 0.6:
            probability *= 0.5
        probability = max(0.02, min(0.95, probability))
        return random.random() < probability

    def get_emotional_dp_prompt(self, full_obs, impression, current_state):
        """生成情绪化的决策提示词"""
        emotion_desc = self.get_emotion_description()

        if current_state == 'MainState':
            prompt = f"""
    请你扮演以下角色:
    [人物特点] {self.profile['description']}
    [当前情绪] {emotion_desc}
    [心理状况] 愤怒：{self.emotion_vector['anger']:.1f}, 
               焦虑：{self.emotion_vector['anxiety']:.1f},
               信任：{self.emotion_vector['trust']:.1f}

    你的印象是: {impression}

    基于你当前激动的情绪状态，选择最想做的行动：
    [0] 点赞（表示认同）
    [1] 评论（强烈表达你的情绪和观点）
    [2] 转发（让更多人看到）
    [3] 查看更多评论
    [4] 查看某条评论的详情
    [5] 离开（不想看了）

    注意：你现在情绪激动，很想表达自己的想法。

    重要：只输出一个数字（0-5），不要有其他文字。
    输出："""
        else:
            from mechanism import get_dp_prompt
            return get_dp_prompt(self, full_obs, impression, current_state)

        return prompt

    def get_negative_comment_prompt(self, impression, action_type):
        """根据情绪状态生成负面评论的提示词"""
        emotion_desc = self.get_emotion_description()

        if self.emotion_vector['anger'] > 0.7:
            prompt = f"""
你现在{emotion_desc}（愤怒值：{self.emotion_vector['anger']:.1f}）。
对于内容：{impression}

请用愤怒的语气表达你的不满，可以使用强烈的词汇，约30-40字。
不要使用脏话，但要表达出强烈的情绪。
"""
        elif self.emotion_vector['trust'] < 0.3:
            prompt = f"""
你{emotion_desc}（信任度仅：{self.emotion_vector['trust']:.1f}）。
对于内容：{impression}

请表达你的质疑和不信任，暗示事情可能不像表面那样，约30-40字。
可以提出阴谋论或暗示有隐情。
"""
        elif self.emotion_vector['anxiety'] > 0.6:
            prompt = f"""
你{emotion_desc}（焦虑值：{self.emotion_vector['anxiety']:.1f}）。
对于内容：{impression}

请表达你的担忧和恐慌，强调可能的负面后果，约30-40字。
"""
        elif self.emotion_vector['disappointment'] > 0.6:
            prompt = f"""
你{emotion_desc}（失望值：{self.emotion_vector['disappointment']:.1f}）。
对于内容：{impression}

请表达你的失望和无奈，语气消极悲观，约30-40字。
"""
        else:
            prompt = f"""
你当前{emotion_desc}。
对于内容：{impression}

请表达你的负面情绪，可以是批评、质疑或不满，约30-40字。
"""

        return prompt

    def reflection_process(self, full_obs, impression, action_info, reflection_info):
        """反思处理流程，更新记忆和情绪状态，核心逻辑：生成反思提示 → LLM评估 → 更新内部状态"""
        action_name = action_info.get('action') if isinstance(action_info, dict) else str(action_info)
        if self._is_lightweight_mode_enabled() and action_name not in {'Comment', 'Reply'}:
            self._lightweight_reflection_process(impression, action_info, reflection_info)
            return

        combined_prompt = self.get_combined_reflection_prompt(impression, action_info)
        reflection_info['prompt'] = {
            'combined_prompt': combined_prompt
        }

        combined_result = None
        parsed_combined = None
        try:
            combined_result = self.llm.fast_run(combined_prompt)
            parsed_combined = self.parse_combined_reflection_result(combined_result)
        except Exception as e:
            warnings.warn(f"Combined reflection failed: {e}")

        if parsed_combined is not None:
            reflection_info['response'] = {
                'combined_result': combined_result
            }

            reflection_info['previous_condition'] = {
                'memory': self.memory,
                'emotion': self.emotion,
                'emotion_vector': self.emotion_vector.copy(),
                'social_confidence': self.social_confidence,
                'opinion': self.opinion
            }

            try:
                self.memory = parsed_combined['summary']
                self.opinion = parsed_combined['opinion']
                self.update_emotion_from_reflection(json.dumps(parsed_combined['emotion'], ensure_ascii=False))
                self.social_confidence = parsed_combined['social_confidence']
                self.emotion_vector['trust'] = self.social_confidence

                reflection_info['after_condition'] = {
                    'memory': self.memory,
                    'emotion': self.emotion,
                    'emotion_vector': self.emotion_vector.copy(),
                    'social_confidence': self.social_confidence,
                    'opinion': self.opinion
                }
                return
            except Exception as e:
                warnings.warn(f"Combined reflection parse/apply failed: {e}")

        from mechanism import get_rp_prompt_summary, get_rp_prompt_opinion, get_rp_prompt_socialconf
        rp_prompt_summary = get_rp_prompt_summary(self, impression, action_info)
        rp_prompt_opinion = get_rp_prompt_opinion(self, impression, action_info)
        rp_prompt_emotion = self.get_rp_prompt_multidim_emotion(impression, action_info)
        rp_prompt_socialconf = get_rp_prompt_socialconf(self, impression, action_info)

        reflection_info['prompt'] = {
            'prompt_summary': rp_prompt_summary,
            'prompt_opinion': rp_prompt_opinion,
            'prompt_emotion': rp_prompt_emotion,
            'prompt_socialconf': rp_prompt_socialconf
        }

        result_summary = self.llm.fast_run(rp_prompt_summary)
  #      print('(memory):', result_summary)
        result_opinion = self.llm.fast_run(rp_prompt_opinion)
  #      print('(opinion):', result_opinion)
        result_emotion = self.llm.fast_run(rp_prompt_emotion)
  #      print('(emotion):', result_emotion)
        result_socialconf = self.llm.fast_run(rp_prompt_socialconf)
  #      print('(social confidence):', result_socialconf)

        reflection_info['response'] = {
            'result_summary': result_summary,
            'result_opinion': result_opinion,
            'result_emotion': result_emotion,
            'result_socialconf': result_socialconf
        }


        reflection_info['previous_condition'] = {
            'memory': self.memory,
            'emotion': self.emotion,
            'emotion_vector': self.emotion_vector.copy(),
            'social_confidence': self.social_confidence,
            'opinion': self.opinion
        }

        try:
            self.memory = result_summary
            self.opinion = result_opinion

            self.update_emotion_from_reflection(result_emotion)

            if len(re.findall('\d+', result_socialconf)) > 0:
                self.social_confidence = min(float(re.findall('\d+', result_socialconf)[0]) / 100.0, 1.0)
                self.emotion_vector['trust'] = self.social_confidence

            reflection_info['after_condition'] = {
                'memory': self.memory,
                'emotion': self.emotion,
                'emotion_vector': self.emotion_vector.copy(),
                'social_confidence': self.social_confidence,
                'opinion': self.opinion
            }

        except Exception as e:
            reflection_info['after'] = '[REFLECTION_EXCEPTION]'
            warnings.warn(str(e))

    def _lightweight_reflection_process(self, impression, action_info, reflection_info):
        """轻量反思：浏览类动作仅做有限状态更新，不调用LLM。"""
        action_name = action_info.get('action', '') if isinstance(action_info, dict) else str(action_info)

        reflection_info['prompt'] = {'mode': 'lightweight'}
        reflection_info['previous_condition'] = {
            'memory': self.memory,
            'emotion': self.emotion,
            'emotion_vector': self.emotion_vector.copy(),
            'social_confidence': self.social_confidence,
            'opinion': self.opinion
        }

        topic = self._extract_topic_snippet(impression)
        self.memory = f"我刚刚浏览了“{topic}”，采取了{action_name}。"

        if action_name in {'Like tweet', 'Like comment'}:
            self.social_confidence = min(1.0, self.social_confidence + 0.01)
            self.emotion_vector['trust'] = min(1.0, self.emotion_vector['trust'] + 0.01)
        elif action_name in {'More Comments', 'Detailed Comment'}:
            self.emotion_vector['anxiety'] = min(1.0, self.emotion_vector['anxiety'] + 0.01)
        elif action_name in {'Leave', 'Back'}:
            for emotion in ['anger', 'anxiety', 'disappointment']:
                self.emotion_vector[emotion] = max(0.0, self.emotion_vector[emotion] - 0.01)

        if action_name in {'Comment', 'Reply'}:
            self._update_overall_positivity()
        else:
            self.emotion = self.emotion_vector['positivity']
        reflection_info['response'] = {'mode': 'lightweight', 'memory': self.memory}
        reflection_info['after_condition'] = {
            'memory': self.memory,
            'emotion': self.emotion,
            'emotion_vector': self.emotion_vector.copy(),
            'social_confidence': self.social_confidence,
            'opinion': self.opinion
        }

    def get_combined_reflection_prompt(self, impression, action_info):
        return f"""
请你扮演以下角色:
[人物特点] {self.profile['description']}
[人物记忆] {self.memory}
[人物观点] {self.opinion}
[当前社会信心] {self.social_confidence:.3f}
[当前情绪]
- anger: {self.emotion_vector['anger']:.3f}
- anxiety: {self.emotion_vector['anxiety']:.3f}
- disappointment: {self.emotion_vector['disappointment']:.3f}
- trust: {self.emotion_vector['trust']:.3f}
- positivity: {self.emotion_vector['positivity']:.3f}

你刚刚的印象是: {impression}
你刚刚执行的动作是: {action_info}

请一次性输出新的反思结果，并严格只返回JSON：
{{
  "summary": "不超过80字的新记忆总结",
  "opinion": "不超过80字的新观点",
  "social_confidence": 0.x,
  "emotion": {{
    "anger": 0.x,
    "anxiety": 0.x,
    "disappointment": 0.x
  }}
}}
"""

    def parse_combined_reflection_result(self, result):
        if not result:
            return None

        json_match = re.search(r'\{.*\}', result, re.S)
        if not json_match:
            return None

        data = json.loads(json_match.group())
        if 'summary' not in data or 'opinion' not in data or 'social_confidence' not in data:
            return None

        emotion = data.get('emotion', {})
        parsed = {
            'summary': str(data['summary']),
            'opinion': str(data['opinion']),
            'social_confidence': max(0.0, min(1.0, float(data['social_confidence']))),
            'emotion': {
                'anger': max(0.0, min(1.0, float(emotion.get('anger', self.emotion_vector['anger'])))),
                'anxiety': max(0.0, min(1.0, float(emotion.get('anxiety', self.emotion_vector['anxiety'])))),
                'disappointment': max(0.0, min(1.0, float(emotion.get('disappointment', self.emotion_vector['disappointment']))))
            }
        }
        return parsed

    def get_rp_prompt_multidim_emotion(self, impression, action_info):
        """生成多维情绪反思的提示词"""
        prompt = f"""
请你扮演以下角色:
[人物特点] {self.profile['description']}
[当前情绪状态] 
- 愤怒：{self.emotion_vector['anger']:.1f}
- 焦虑：{self.emotion_vector['anxiety']:.1f}
- 失望：{self.emotion_vector['disappointment']:.1f}
- 信任：{self.emotion_vector['trust']:.1f}

你刚刚的印象是: {impression}
你执行了动作：{action_info}

请评估你执行动作后的情绪变化。输出JSON格式的新情绪值（0-1范围）：
{{"anger": x.x, "anxiety": x.x, "disappointment": x.x}}

只输出JSON，不要其他解释。
"""
        return prompt

    def update_emotion_from_reflection(self, emotion_result):
        """从反思结果更新多维情绪：混合新旧值避免剧烈波动"""
        try:
            import json
            json_match = re.search(r'\{[^}]+\}', emotion_result)
            if json_match:
                emotion_dict = json.loads(json_match.group())

                for emotion in ['anger', 'anxiety', 'disappointment']:
                    if emotion in emotion_dict:
                        new_value = float(emotion_dict[emotion])
                        old_value = self.emotion_vector[emotion]
                        self.emotion_vector[emotion] = 0.7 * new_value + 0.3 * old_value
                        self.emotion_vector[emotion] = max(0.0, min(1.0,
                                                                    self.emotion_vector[emotion]))

        except Exception as e:
            print(f"情绪更新失败: {e}")
            for emotion in ['anger', 'anxiety', 'disappointment']:
                self.emotion_vector[emotion] *= 0.95

        self._update_overall_positivity()

    def _update_overall_positivity(self, smooth=True):
        """更新整体积极性：用保守公式避免高信任用户被直接推到满积极。"""
        anger = self.emotion_vector.get('anger', 0)
        anxiety = self.emotion_vector.get('anxiety', 0)
        disappointment = self.emotion_vector.get('disappointment', 0)
        trust = self.emotion_vector.get('trust', 0)

        negative_sum = (anger * 0.4 + anxiety * 0.3 + disappointment * 0.3)
        target_positivity = 0.7 * (1.0 - negative_sum) + 0.3 * trust
        current_positivity = self.emotion_vector.get('positivity', 0.5)

        if smooth:
            positivity = 0.6 * current_positivity + 0.4 * target_positivity
        else:
            positivity = target_positivity

        self.emotion_vector['positivity'] = max(0.0, min(1.0, positivity))
        self.emotion = self.emotion_vector['positivity']

    def take_action(self, full_obs, current_state, agent_action_info):
        """实现完整的智能代理决策循环：感知 → 决策 → 反思"""
        if hasattr(self, 'simulator') and self.simulator and hasattr(self.simulator.time_system, 'current_time'):
            self.last_active_time = self.simulator.time_system.current_time

        sensory_info = {}
        impression = self.sensory_process(full_obs, sensory_info)
        print('(impression):', impression)

        decision_info = {}
        action, text_response, action_info = self.decision_process(
            full_obs, impression, current_state, decision_info)

        if isinstance(action_info, dict):
            action_info['timestamp'] = time.time()

        reflection_info = {}
        self.reflection_process(full_obs, impression, action_info, reflection_info)

        agent_action_info['sensory_info'] = sensory_info
        agent_action_info['decision_info'] = decision_info
        agent_action_info['reflection_info'] = reflection_info
        agent_action_info['emotion_state'] = self.emotion_vector.copy()

        return action, text_response, action_info

    def get_agent_summary(self):
        """获取代理的摘要信息（用于分析）"""
        return {
            'aid': self.aid,
            'name': self.name,
            'type': self.type,
            'emotion_vector': self.emotion_vector.copy(),
            'negative_behavior_tendency': self.negative_behavior_tendency,
            'behavior_count': len(self.behavior_history),
            'last_active': self.last_active_time
        }
