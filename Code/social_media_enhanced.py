import re
import time
from typing import Dict, List, Optional
from collections import defaultdict

from mechanism import *
from utils_enhanced import *
from emotion_analyzer import EmotionAnalyzer

class Reply():
    """评论回复数据模型：存储单条回复的基本信息和情绪特征"""
    def __init__(self, rid, cid, reply_text, author_id=None):
        self.rid = rid
        self.cid = cid
        self.reply_text = reply_text
        self.author_id = author_id
        self.timestamp = time.time()
        self.emotion_scores = {}

    def __str__(self):
        return self.reply_text


class Comment():
    def __init__(self, cid, comment_text, author_id=None):
        self.cid = cid
        self.comment_text = comment_text
        self.author_id = author_id

        self.like_num = 0
        self.reply_list = []
        self.timestamp = time.time()

        self.emotion_scores = {}
        self.visibility_weight = 1.0
        self.is_negative = False
        self.is_intervention = False

    def __str__(self):
        return self.comment_text

    def __lt__(self, other):
        if isinstance(other, Comment):
            self_score = self.like_num * self.visibility_weight
            other_score = other.like_num * other.visibility_weight
            return self_score < other_score


class SocialMedia():
    """
    社交平台核心类：模拟社交媒体环境，管理用户评论、情绪分析、内容过滤和互动行为
    支持干预机制（如辟谣、情绪安抚）和动态内容展示逻辑
    """
    def __init__(self, config, llm):
        self.global_config = config
        self.config = config['media_config']
        self.llm = llm

        self.tweet_info = load_tweet(config)

        self.comment_list = SortedList(Comment)
        self.filtered_comments = []
        self.detect_list = []

        self.view_num = 0
        self.like_num = 0
        self.repost_num = 0

        self.emotion_analyzer = EmotionAnalyzer(llm)

        self.content_stats = {
            'total_comments': 0,
            'negative_comments': 0,
            'positive_comments': 0,
            'neutral_comments': 0,
            'intervention_comments': 0,
            'filtered_comments': 0,
            'downranked_comments': 0
        }

        self.emotion_history = []

        self.interaction_network = defaultdict(list)

        self.positive_content_boost = config.get('intervention', {}).get('positive_content', {})

    def recover_llm(self, llm):
        self.llm = llm
        self.emotion_analyzer = EmotionAnalyzer(llm)

    def analyze_and_categorize_comment(self, text, author_id=None, is_intervention=False):
        """对评论文本进行情绪分析，判断正负性"""
        emotion_scores = self.emotion_analyzer.analyze_content_emotion(text)

        negative_indicators = {
            'anger': emotion_scores.get('anger', 0) * 0.35,
            'anxiety': emotion_scores.get('anxiety', 0) * 0.25,
            'disappointment': emotion_scores.get('disappointment', 0) * 0.2,
            'negativity': emotion_scores.get('negativity', 0) * 0.2
        }

        negative_score = sum(negative_indicators.values())

        if 'positivity' not in emotion_scores:
            negative_average = (
                emotion_scores.get('anger', 0) +
                emotion_scores.get('anxiety', 0) +
                emotion_scores.get('disappointment', 0) +
                emotion_scores.get('negativity', 0)
            ) / 4
            emotion_scores['positivity'] = max(0.0, min(1.0, 1.0 - negative_average))

        text_features = self.extract_text_features(text)
        if text_features['has_strong_negative_words']:
            negative_score += 0.2
        if text_features['has_excessive_punctuation']:
            negative_score += 0.1
        if text_features['has_caps_lock']:
            negative_score += 0.1

        threshold = self.global_config.get('analysis_config', {}).get('negative_comment_threshold', 0.45)

        positivity = emotion_scores.get('positivity', 0.5)
        negativity = emotion_scores.get('negativity', 0)
        max_negative_emotion = max(
            emotion_scores.get('anger', 0),
            emotion_scores.get('anxiety', 0),
            emotion_scores.get('disappointment', 0)
        )

        is_negative = negative_score >= threshold and positivity < 0.65
        is_positive = (
            not is_negative and
            positivity >= 0.62 and
            negativity < 0.35 and
            max_negative_emotion < 0.4
        )

        if is_negative:
            self.content_stats['negative_comments'] += 1
        elif is_positive:
            self.content_stats['positive_comments'] += 1
        else:
            self.content_stats['neutral_comments'] += 1

        if is_intervention:
            self.content_stats['intervention_comments'] += 1

        return emotion_scores, is_negative

    def extract_text_features(self, text):
        """【新增】：提取文本特征用于辅助情绪判断"""
        features = {
            'has_strong_negative_words': False,
            'has_excessive_punctuation': False,
            'has_caps_lock': False,
            'length': len(text)
        }

        strong_negative_words = [
            '垃圾', '恶心', '可恶', '该死', '去死', '滚',
            '骗子', '无耻', '恶毒', '败类', '人渣'
        ]
        for word in strong_negative_words:
            if word in text:
                features['has_strong_negative_words'] = True
                break

        import re
        if len(re.findall(r'[!！]{3,}|[?？]{3,}', text)) > 0:
            features['has_excessive_punctuation'] = True

        english_words = re.findall(r'[A-Z]{3,}', text)
        if len(english_words) > 2:
            features['has_caps_lock'] = True

        return features

    def _evaluate_content_filter(self, text, emotion_scores, current_time, is_intervention):
        """评估内容过滤结果；干预内容豁免过滤，但不改写情绪观测值。"""
        result = {
            'should_filter': False,
            'filter_reason': None,
            'visibility_weight': 1.0,
            'downranked': False,
            'weighted_score': 0.0
        }

        if is_intervention:
            return result

        filter_config = self.global_config['intervention']['content_filtering']
        if filter_config.get('enabled', False):
            filter_dimensions = filter_config.get('filter_dimensions', {})
            filter_scores = {
                'anger': emotion_scores.get('anger', 0),
                'anxiety': emotion_scores.get('anxiety', 0),
                'negativity': emotion_scores.get('negativity', 0),
                'misinformation': emotion_scores.get('misinformation', 0)
            }

            weighted_score = (
                    filter_scores['anger'] * 0.3 +
                    filter_scores['anxiety'] * 0.2 +
                    filter_scores['negativity'] * 0.35 +
                    filter_scores['misinformation'] * 0.15
            )
            result['weighted_score'] = weighted_score

            if weighted_score > filter_config.get('negative_threshold', 0.5):
                result['visibility_weight'] = filter_config.get('visibility_reduction', 0.2)
                result['downranked'] = True

                if weighted_score > filter_config.get('severe_threshold', 0.7):
                    result['should_filter'] = True
                    result['filter_reason'] = f"严重负面内容 (得分: {weighted_score:.2f})"

                for dim, threshold in filter_dimensions.items():
                    if filter_scores.get(dim, 0) > threshold:
                        result['should_filter'] = True
                        result['filter_reason'] = f"{dim}超标 ({filter_scores[dim]:.2f} > {threshold})"
                        break

        if not result['should_filter']:
            split_time = self.global_config['intervention']['comment_poisoning_detect']['split_time']
            threshold_time = (1.0 - split_time) * timestring_to_timestamp(
                self.global_config['time_config']['start_time']) + split_time * \
                             timestring_to_timestamp(self.global_config['time_config']['end_time'])

            if (self.global_config['intervention']['comment_poisoning_detect']['type'] == 'LLM'
                    and current_time >= threshold_time
                    and result['weighted_score'] > 0.4):
                cleaned_text = self.prepare_text_for_api(text)
                prompt = f"""评论内容：{cleaned_text}

    请判断这条评论是否包含以下恶意内容：
    1. 仇恨言论或人身攻击
    2. 煽动性言论
    3. 明显的虚假信息
    4. 恶意引战内容

    如果包含上述任何内容，回答1；否则回答0。
    只输出数字，不要解释。"""

                try:
                    llm_result = self.llm.fast_run(prompt)
                    code_list = re.findall('\d', llm_result)
                    if len(code_list) > 0 and code_list[0] == '1':
                        self.detect_list.append(text)
                        result['should_filter'] = True
                        result['filter_reason'] = "LLM检测为恶意内容"
                except Exception as e:
                    print(f"恶意检测失败: {e}")

        if result['should_filter']:
            result['downranked'] = False

        return result

    def action_response(self, action, text_response, action_info, current_comment_id,
                        current_time, agent_id=None, is_intervention=False):
        """动作响应分发：将用户动作路由到对应的处理方法"""
        if isinstance(action_info, dict) and 'timestamp' in action_info:
            action_info['timestamp'] = current_time

        if action == 'Like tweet':
            action = 'Like_tweet'
        if action == 'More Comments':
            action = 'More_Comments'
        if action == 'Detailed Comment':
            action = 'Detailed_Comment'

        method_name = f'{action}_action'
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            result = method(text_response, action_info, current_comment_id,
                            current_time, agent_id, is_intervention)

            if is_intervention and hasattr(result, 'visibility_weight'):
                result.visibility_weight = 3.0
            return result
        else:
            print(f"Warning: Unknown action {action}")
            return 'MainState'

    def Leave_action(self, text_response, action_info, current_comment_id,
                     current_time, agent_id=None, is_intervention=False):
        return 'EndState'

    def NoAction_action(self, text_response, action_info, current_comment_id,
                        current_time, agent_id=None, is_intervention=False):
        """处理无操作指令（消除Unknown action警告）"""
        if self.global_config.get('debug', False):
            print(f"[NoAction] Agent {agent_id} took no action at {current_time}")
        return 'MainState'

    def Enter_action(self, text_response, action_info, current_comment_id,
                     current_time, agent_id=None, is_intervention=False):
        self.view_num += 1
        return 'MainState'

    def get_comment_author_id(self, comment):
        """获取评论作者ID"""
        """获取评论作者ID，确保不返回None"""
        if hasattr(comment, 'author_id') and comment.author_id is not None:
           return comment.author_id
        else:
            print(f"Warning: Comment (CID: {comment.cid}) has no author, using default ID '0'")
            return 0

    def get_comment_by_cid(self, cid):
        """根据稳定CID获取评论对象，避免使用动态排序索引"""
        if cid is None or cid < 0 or cid >= len(self.comment_list.loc):
            return None

        location = self.comment_list.loc[cid]
        if location is None or location >= len(self.comment_list.sorted):
            return None

        comment = self.comment_list.sorted[location]
        if getattr(comment, 'cid', None) == cid:
            return comment

        for comment in self.comment_list.sorted:
            if getattr(comment, 'cid', None) == cid:
                return comment
        return None

    def get_visible_comment_map(self, page=0, consider_weight=True):
        """返回当前页可见评论与CID的映射。"""
        visible_comments = self.get_visible_comments(page=page, consider_weight=consider_weight)
        return [
            {
                'display_index': index,
                'cid': comment.cid,
                'comment': comment
            }
            for index, comment in enumerate(visible_comments)
        ]

    def get_visible_comment_ids(self, page=0, consider_weight=True):
        return [
            entry['cid']
            for entry in self.get_visible_comment_map(page=page, consider_weight=consider_weight)
        ]

    def resolve_comment_id(self, comment_ref, visible_comment_ids=None):
        """将界面索引或CID统一解析为稳定CID"""
        if isinstance(comment_ref, str):
            try:
                comment_ref = int(comment_ref)
            except ValueError:
                return None

        if visible_comment_ids and isinstance(comment_ref, int) and 0 <= comment_ref < len(visible_comment_ids):
            return visible_comment_ids[comment_ref]

        if isinstance(comment_ref, int) and self.get_comment_by_cid(comment_ref) is not None:
            return comment_ref

        return None

    def Like_tweet_action(self, text_response, action_info, current_comment_id,
                          current_time, agent_id=None, is_intervention=False):
        self.like_num += 1
        return 'MainState'

    def Comment_action(self, text_response, action_info, current_comment_id,
                       current_time, agent_id=None, is_intervention=False):
        """处理"评论"动作：创建评论对象，执行内容过滤，更新统计
        核心逻辑：文本清理→情绪分析→多层过滤→添加评论/过滤列表"""
        text_response = self.enhanced_sanitize_text(text_response)

        emotion_scores, is_negative = self.analyze_and_categorize_comment(
            text_response, agent_id, is_intervention)
        if isinstance(action_info, dict):
            action_info['generated_content_emotion_scores'] = emotion_scores
            action_info['generated_content_is_negative'] = is_negative

        comment = Comment(len(self.comment_list), text_response, agent_id)
        comment.emotion_scores = emotion_scores
        comment.is_negative = is_negative
        comment.is_intervention = is_intervention

        filter_result = self._evaluate_content_filter(
            text_response, emotion_scores, current_time, is_intervention)
        should_filter = filter_result['should_filter']
        filter_reason = filter_result['filter_reason']
        comment.visibility_weight = filter_result['visibility_weight']
        if filter_result['downranked']:
            self.content_stats['downranked_comments'] = self.content_stats.get('downranked_comments', 0) + 1

        if not should_filter:
            if is_intervention or emotion_scores.get('positivity', 0) > 0.7:
                positive_boost = self.global_config['intervention'].get('positive_content', {})
                if positive_boost.get('enabled', False):
                    comment.visibility_weight = positive_boost.get('boost_factor', 2.0)
                    comment.like_num = random.randint(1, positive_boost.get('initial_likes', 3))

        if not should_filter:
            self.comment_list.append(comment.cid, comment.comment_text)
            added_comment = self.comment_list.sorted[-1]
            added_comment.emotion_scores = emotion_scores
            added_comment.is_negative = is_negative
            added_comment.is_intervention = is_intervention
            added_comment.visibility_weight = comment.visibility_weight
            added_comment.author_id = agent_id
            added_comment.like_num = comment.like_num
        else:
            print(f"[内容过滤] {filter_reason}")
            self.content_stats['filtered_comments'] += 1
            self.filtered_comments.append({
                'comment': comment,
                'reason': filter_reason,
                'timestamp': current_time
            })

        self.content_stats['total_comments'] += 1

        self.emotion_history.append({
            'timestamp': current_time,
            'comment_id': comment.cid,
            'emotion_scores': emotion_scores,
            'is_negative': is_negative,
            'is_intervention': is_intervention,
            'author_id': agent_id,
            'filtered': should_filter,
            'filter_reason': filter_reason
        })

        return 'MainState'

    def Repost_action(self, text_response, action_info, current_comment_id,
                      current_time, agent_id=None, is_intervention=False):
        self.repost_num += 1
        return 'MainState'

    def More_Comments_action(self, text_response, action_info, current_comment_id,
                             current_time, agent_id=None, is_intervention=False):
        num_new_comments = 3
        start_idx = max(0, len(self.comment_list) - num_new_comments)
        new_comments = list(self.comment_list)[start_idx:]

        formatted_comments = []
        for i, comment in enumerate(new_comments):
            formatted_comments.append(f"[{i + 1}] {comment.comment_text} (点赞:{comment.like_num})")

        if formatted_comments:
            action_info['extra_comments'] = '\n'.join(formatted_comments)
            action_info['has_new_comments'] = True
        else:
            action_info['extra_comments'] = "暂无更多评论"
            action_info['has_new_comments'] = False

        return 'MainState'

    def Detailed_Comment_action(self, text_response, action_info, current_comment_id,
                                current_time, agent_id=None, is_intervention=False):
        visible_comment_ids = action_info.get('visible_comment_ids', []) if isinstance(action_info, dict) else []
        target_comment_id = self.resolve_comment_id(text_response, visible_comment_ids)
        if isinstance(action_info, dict):
            action_info['target_comment_id'] = target_comment_id

        if target_comment_id is None:
            return 'MainState'

        target_comment = self.get_comment_by_cid(target_comment_id)
        if agent_id is not None and target_comment is not None and target_comment.author_id is not None:
            self.interaction_network[agent_id].append(target_comment.author_id)

        return 'CommentState'

    def Like_comment_action(self, text_response, action_info, current_comment_id,
                            current_time, agent_id=None, is_intervention=False):
        target_comment = self.get_comment_by_cid(current_comment_id)
        if target_comment is None:
            return 'CommentState'

        self.comment_list.add_like(target_comment.cid)

        if target_comment.is_intervention:
            self.comment_list.add_like(target_comment.cid)

        if agent_id is not None and target_comment.author_id is not None:
            self.interaction_network[agent_id].append(target_comment.author_id)

        return 'CommentState'

    def Reply_action(self, text_response, action_info, current_comment_id,
                     current_time, agent_id=None, is_intervention=False):
        """处理回复动作：创建回复对象，添加到评论的回复列表"""
        try:
            target_comment = self.get_comment_by_cid(current_comment_id)
            if target_comment is not None:
                if text_response:
                    text_response = self.enhanced_sanitize_text(text_response)

                emotion_scores, is_negative = self.analyze_and_categorize_comment(
                    text_response, agent_id, is_intervention)
                if isinstance(action_info, dict):
                    action_info['generated_content_emotion_scores'] = emotion_scores
                    action_info['generated_content_is_negative'] = is_negative

                reply = Reply(
                    len(target_comment.reply_list),
                    target_comment.cid,
                    text_response,
                    agent_id
                )
                reply.emotion_scores = emotion_scores
                reply.is_negative = is_negative
                reply.is_intervention = is_intervention

                filter_result = self._evaluate_content_filter(
                    text_response, emotion_scores, current_time, is_intervention)
                should_filter = filter_result['should_filter']
                filter_reason = filter_result['filter_reason']
                reply.visibility_weight = filter_result['visibility_weight']
                reply.filtered = should_filter

                if filter_result['downranked']:
                    self.content_stats['downranked_comments'] = self.content_stats.get('downranked_comments', 0) + 1

                if should_filter:
                    print(f"[回复过滤] {filter_reason}")
                    self.content_stats['filtered_comments'] += 1
                    self.filtered_comments.append({
                        'reply': reply,
                        'comment_id': target_comment.cid,
                        'reason': filter_reason,
                        'timestamp': current_time
                    })
                else:
                    target_comment.reply_list.append(reply)

                self.content_stats['total_comments'] += 1
                self.emotion_history.append({
                    'timestamp': current_time,
                    'comment_id': target_comment.cid,
                    'reply_id': reply.rid,
                    'content_type': 'reply',
                    'emotion_scores': emotion_scores,
                    'is_negative': is_negative,
                    'is_intervention': is_intervention,
                    'author_id': agent_id,
                    'filtered': should_filter,
                    'filter_reason': filter_reason
                })

                if agent_id is not None and target_comment.author_id is not None:
                    self.interaction_network[agent_id].append(target_comment.author_id)

        except Exception as e:
            print(f'Reply fails: {e}')

        return 'CommentState'

    def enhanced_sanitize_text(self, text):
        """增强的文本清理，避免API错误"""
        if not text:
            return ""

        # 移除或替换特殊字符
        import re

        text = ''.join(char for char in text if ord(char) >= 32 and ord(char) < 127 or ord(char) > 127)

        text = re.sub(r'\{([^}]*)\}', r'（\1）', text)

        placeholders = [
            (r'{context}', '这件事'),
            (r'{point}', '这种观点'),
            (r'{doubt}', '疑点'),
            (r'{alternative}', '另一种可能'),
            (r'{conspiracy}', '背后原因'),
            (r'{rage}', '这种情况'),
            (r'{panic}', '这个问题'),
            (r'{despair}', '现状')
        ]

        for pattern, replacement in placeholders:
            text = text.replace(pattern, replacement)

        text = re.sub(r'[!！]{4,}', '！！！', text)
        text = re.sub(r'[?？]{4,}', '？？？', text)
        text = re.sub(r'\.{4,}', '...', text)
        text = re.sub(r'。{4,}', '。。。', text)

        text = re.sub(r'(.)\1{5,}', r'\1\1\1', text)

        sensitive_replacements = [
            ('死', '逝'),
            ('杀', '害'),
            ('爆', '炸'),
            ('黑社会', '不法组织'),
            ('贪污', '腐败')
        ]

        for old, new in sensitive_replacements:
            text = text.replace(old, new)

        if len(text) > 150:
            text = text[:147] + "..."

        return text.strip()

    def prepare_text_for_api(self, text):
        """为API调用准备文本，移除可能导致问题的内容"""
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[链接]',
                      text)

        text = re.sub(r'@\w+', '[用户]', text)

        text = re.sub(r'#\w+', '', text)

        text = self.enhanced_sanitize_text(text)

        return text

    def Back_action(self, text_response, action_info, current_comment_id,
                    current_time, agent_id=None, is_intervention=False):
        return 'MainState'

    def get_current_state(self):
        """获取当前社交媒体状态（供干预代理使用）"""
        recent_emotions = self.emotion_history[-20:] if len(self.emotion_history) > 20 else self.emotion_history

        avg_negative_emotion = 0
        intervention_count = 0
        if recent_emotions:
            for entry in recent_emotions:
                scores = entry['emotion_scores']
                negative = (scores.get('anger', 0) + scores.get('anxiety', 0) +
                            scores.get('disappointment', 0)) / 3
                avg_negative_emotion += negative
                if entry.get('is_intervention', False):
                    intervention_count += 1
            avg_negative_emotion /= len(recent_emotions)

        negative_ratio = min(1.0, self.content_stats['negative_comments'] / max(1, self.content_stats['total_comments']))

        return {
            'recent_comments': [str(c) for c in self.comment_list.sorted[-5:]],
            'total_comments': self.content_stats['total_comments'],
            'avg_negative_emotion': avg_negative_emotion,
            'negative_comment_ratio': negative_ratio,
            'filtered_count': self.content_stats['filtered_comments'],
            'intervention_count': self.content_stats['intervention_comments'],
            'intervention_ratio': intervention_count / max(1, len(recent_emotions))
        }

    def get_visible_comments(self, page=0, consider_weight=True):
        """获取可见评论，考虑权重和干预优先级，支持分页"""
        start = page * self.config['comment_pagesize']
        end = start + self.config['comment_pagesize']

        if not consider_weight:
            return self.comment_list.sorted[start:end]

        def comment_score(c):
            base_score = c.like_num * c.visibility_weight
            if c.is_intervention:
                base_score *= 2.0
            recency_bonus = 1.0 + (0.1 if hasattr(c, 'timestamp') and
                                          time.time() - c.timestamp < 300 else 0)
            return base_score * recency_bonus

        weighted_comments = sorted(
            self.comment_list.sorted,
            key=comment_score,
            reverse=True
        )

        return weighted_comments[start:end]


    def get_emotion_statistics(self):
        """获取情绪统计信息：时间序列情绪数据、平均值、峰值、干预效果等"""
        if not self.emotion_history:
            return {}

        timeline_data = []
        for entry in self.emotion_history:
            timeline_data.append({
                'timestamp': entry['timestamp'],
                'anger': entry['emotion_scores'].get('anger', 0),
                'anxiety': entry['emotion_scores'].get('anxiety', 0),
                'disappointment': entry['emotion_scores'].get('disappointment', 0),
                'is_negative': entry['is_negative'],
                'is_intervention': entry.get('is_intervention', False)
            })

        total_anger = 0
        total_anxiety = 0
        total_disappointment = 0
        negative_count = 0

        for e in timeline_data:
            total_anger += e['anger']
            total_anxiety += e['anxiety']
            total_disappointment += e['disappointment']
            if e['is_negative']:
                negative_count += 1

        n = max(1, len(timeline_data))

        return {
            'timeline': timeline_data,
            'averages': {
                'anger': total_anger / n,
                'anxiety': total_anxiety / n,
                'disappointment': total_disappointment / n,
                'negative_ratio': negative_count / n
            },
            'peaks': {
                'max_anger': max((e['anger'] for e in timeline_data), default=0),
                'max_anxiety': max((e['anxiety'] for e in timeline_data), default=0),
                'max_disappointment': max((e['disappointment'] for e in timeline_data), default=0)
            },
            'intervention_effectiveness': 0.0
        }

    def get_interaction_network_stats(self):
        """获取互动网络统计：用户互动次数、网络密度、最活跃用户等"""
        interaction_counts = {
            agent_id: len(interactions)
            for agent_id, interactions in self.interaction_network.items()
        }

        if interaction_counts:
            most_active = max(interaction_counts.items(), key=lambda x: x[1])
        else:
            most_active = (None, 0)

        total_possible = len(interaction_counts) * (len(interaction_counts) - 1)
        actual_interactions = sum(interaction_counts.values())
        density = actual_interactions / total_possible if total_possible > 0 else 0

        return {
            'interaction_counts': interaction_counts,
            'most_active_agent': most_active,
            'network_density': density,
            'total_interactions': actual_interactions
        }

    def get_current_info(self):
        """获取当前完整信息（用于模拟结束时记录）：评论、统计、过滤内容等"""
        processed_filtered_comments = []
        for cmt in self.filtered_comments:
            if isinstance(cmt, dict):
                if 'reply' in cmt:
                    reply_obj = cmt['reply']
                    processed_filtered_comments.append({
                        'content': reply_obj.reply_text if hasattr(reply_obj, 'reply_text') else str(reply_obj),
                        'emotion_scores': reply_obj.emotion_scores if hasattr(reply_obj, 'emotion_scores') else {},
                        'reason': cmt.get('reason', ''),
                        'timestamp': cmt.get('timestamp', ''),
                        'content_type': 'reply',
                        'comment_id': cmt.get('comment_id')
                    })
                elif 'comment' in cmt:
                    comment_obj = cmt['comment']
                    processed_filtered_comments.append({
                        'content': comment_obj.comment_text if hasattr(comment_obj, 'comment_text') else str(
                            comment_obj),
                        'emotion_scores': comment_obj.emotion_scores if hasattr(comment_obj, 'emotion_scores') else {},
                        'reason': cmt.get('reason', ''),
                        'timestamp': cmt.get('timestamp', ''),
                        'content_type': 'comment'
                    })
                else:
                    processed_filtered_comments.append({
                        'content': cmt.get('comment_text', ''),
                        'emotion_scores': cmt.get('emotion_scores', {}),
                        'reason': cmt.get('reason', ''),
                        'timestamp': cmt.get('timestamp', '')
                    })
            else:
                processed_filtered_comments.append({
                    'content': cmt.comment_text if hasattr(cmt, 'comment_text') else str(cmt),
                    'emotion_scores': cmt.emotion_scores if hasattr(cmt, 'emotion_scores') else {},
                    'reason': '',
                    'timestamp': ''
                })

        info_dict = {
            'basic_info': self.tweet_info,
            'comment_list': [{
                'cid': cmt.cid,
                'content': cmt.comment_text,
                'author_id': cmt.author_id,
                'like_num': cmt.like_num,
                'emotion_scores': cmt.emotion_scores,
                'is_negative': cmt.is_negative,
                'is_intervention': cmt.is_intervention,
                'visibility_weight': cmt.visibility_weight,
                'reply_list': cmt.reply_list
            } for cmt in self.comment_list.sorted],
            'filtered_comments': processed_filtered_comments,
            'detect_list': self.detect_list,
            'statistics': {
                'view_num': self.view_num,
                'like_num': self.like_num,
                'repost_num': self.repost_num,
                'content_stats': self.content_stats,
                'emotion_stats': self.get_emotion_statistics(),
                'network_stats': self.get_interaction_network_stats()
            }
        }

        return info_dict
