"""
核心机制模块：包含状态转换、提示词生成、视图构建等核心功能
"""
import numpy as np
from typing import Dict, Any
from utils_enhanced import timestring_to_timestamp


STATE_CODE_TO_ACTION = {
    'EntryState': {
        '0': 'Enter'
    },
    'MainState': {
        '0': 'Like tweet',
        '1': 'Comment',
        '2': 'Repost',
        '3': 'More Comments',
        '4': 'Detailed Comment',
        '5': 'Leave'
    },
    'CommentState': {
        '0': 'Like comment',
        '1': 'Reply',
        '2': 'Back'
    }
}


def get_action_info(action: str, extra_response: Any, timestamp: float = None) -> Dict[str, Any]:
    """获取动作信息（改进版）"""
    import time
    return {
        'action': action,
        'extra_response': extra_response,
        'timestamp': timestamp if timestamp is not None else time.time()
    }


def get_view_info(social_media, current_state: str, comment_page: int = 0,
                  current_comment_id: int = None) -> str:
    """根据当前状态生成视图信息"""

    if current_state == 'EntryState':
        tweet = social_media.tweet_info
        view_info = f"""推文标题: {tweet['title']}
来源: {tweet['source']}
发布时间: {tweet['publish_time']}
简介: {tweet['brief_intro']}
"""

    elif current_state == 'MainState':
        tweet = social_media.tweet_info
        view_info = f"""推文内容: {tweet['content']}
点赞数: {social_media.like_num} | 转发数: {social_media.repost_num} | 评论数: {len(social_media.comment_list)}

热门评论:"""

        visible_comments = social_media.get_visible_comment_map(page=comment_page)

        for entry in visible_comments:
            comment = entry['comment']
            i = entry['display_index']
            view_info += f"\n[{i}] {comment.comment_text} (点赞: {comment.like_num})"
            if comment.reply_list:
                view_info += f" | {len(comment.reply_list)}条回复"

    elif current_state == 'CommentState':
        comment = social_media.get_comment_by_cid(current_comment_id)
        if comment is not None:
            view_info = f"""评论详情:
{comment.comment_text}
点赞数: {comment.like_num}

回复列表:"""
            for i, reply in enumerate(comment.reply_list[:4]):
                view_info += f"\n[{i}] {reply}"
        else:
            view_info = "评论不存在"

    else:
        view_info = "未知状态"

    return view_info


def get_sp_prompt(agent, view_info: str) -> str:
    """生成感知处理提示词"""
    return f"""请你扮演以下角色:
[人物特点] {agent.profile['description']}
[人物记忆] {agent.memory}
[人物观点] {agent.opinion}
[当前情绪] {agent.emotion}

你正在浏览社交媒体，刚刚看到的内容是: {view_info}

请以第一人称输出你对这些内容的印象和感受（约40字）："""


def get_dp_prompt(agent, full_obs: str, impression: str, current_state: str) -> str:
    """生成决策提示词"""
    base_prompt = f"""请你扮演以下角色:
[人物特点] {agent.profile['description']}
[人物记忆] {agent.memory}
[人物观点] {agent.opinion}
[社会信心] {agent.social_confidence}
[当前情绪] {agent.emotion}

你刚刚看到的内容是: {full_obs}
你的印象是: {impression}
"""

    if current_state == 'MainState':
        action_prompt = """
基于你的印象，选择一个最想做的行动：
[0] 点赞这条推文
[1] 发表评论
[2] 转发这条推文
[3] 查看更多评论
[4] 查看某条评论的详情
[5] 离开

重要：请只输出一个数字（0-5之间），不要输出其他任何文字。
输出格式：单个数字
示例：1"""

    elif current_state == 'CommentState':
        action_prompt = """
看完这条评论，你想做什么：
[0] 点赞这条评论
[1] 回复这条评论
[2] 返回主页面

重要：请只输出一个数字（0-2之间），不要输出其他任何文字。
输出格式：单个数字
示例：1"""

    else:
        action_prompt = "\n请选择一个动作。"

    return base_prompt + action_prompt


def get_comment_prompt(agent, impression: str) -> str:
    """生成评论提示词"""
    return f"""请你扮演以下角色:
[人物特点] {agent.profile['description']}
[当前观点] {agent.opinion}

基于你的印象: {impression}

请发表一条评论，表达你的观点和感受。要求：
1. 符合你的人物特点
2. 30-50字
3. 真实自然

直接输出评论内容："""


def get_reply_prompt(agent, impression: str) -> str:
    """生成回复提示词"""
    return f"""请你扮演以下角色:
[人物特点] {agent.profile['description']}
[当前观点] {agent.opinion}

你看到的评论给你的印象是: {impression}

请回复这条评论。要求：
1. 符合你的人物特点
2. 20-40字
3. 针对评论内容

直接输出回复内容："""


def get_comment_id(agent, impression: str) -> str:
    """获取要查看的评论ID"""
    return f"""你刚刚看到了一些评论，你的印象是: {impression}

请选择你最想详细查看的评论编号（0-3之间的整数）。
只输出一个数字："""


def get_rp_prompt_summary(agent, impression: str, action_info: Dict) -> str:
    """生成记忆更新提示词"""
    return f"""请你扮演以下角色:
[人物特点] {agent.profile['description']}
[之前的记忆] {agent.memory}

你刚刚的印象是: {impression}
你执行了动作: {action_info}

请用第一人称更新你的记忆，简要总结刚才的经历（约30字）："""


def get_rp_prompt_opinion(agent, impression: str, action_info: Dict) -> str:
    """生成观点更新提示词"""
    return f"""请你扮演以下角色:
[人物特点] {agent.profile['description']}
[之前的观点] {agent.opinion}

你刚刚的印象是: {impression}
你执行了动作: {action_info}

基于新的经历，你的观点有什么变化吗？请用第一人称更新你的观点（约30字）："""


def get_rp_prompt_emotion(agent, impression: str, action_info: Dict) -> str:
    """生成情绪更新提示词"""
    return f"""请你扮演以下角色:
[人物特点] {agent.profile['description']}
[当前情绪] {agent.emotion}

你刚刚的印象是: {impression}
你执行了动作: {action_info}

请评估你现在的情绪状态（0-1之间的小数，0表示非常消极，1表示非常积极）："""


def get_rp_prompt_socialconf(agent, impression: str, action_info: Dict) -> str:
    """生成社会信心更新提示词"""
    return f"""请你扮演以下角色:
[人物特点] {agent.profile['description']}
[当前社会信心] {agent.social_confidence * 100:.0f}%

你刚刚的印象是: {impression}
你执行了动作: {action_info}

基于这次经历，你对社会的信心有什么变化？
请评估你现在的社会信心（0-100的整数）："""


def time_func(x: float, alpha: float, A: float, T: float) -> float:
    """时间概率密度函数"""
    return alpha * A * np.exp(alpha * (A * x - T)) / (1 + np.exp(alpha * (A * x - T))) ** 2
