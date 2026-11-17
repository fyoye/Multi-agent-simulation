"""
LLM接口模块：提供与大语言模型的交互接口（完整修改版）
"""
import requests
import json
import time
from typing import Dict, Any, Optional
import random
import re


class LLM:
    """LLM基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name', 'gpt-3.5-turbo')
        self.api_key = config.get('API_KEY', '')
        self.max_retries = 3
        self.retry_delay = 1.0
        self.use_mock = config.get('use_mock', False)

    def fast_run(self, prompt: str) -> str:
        """快速运行，获取模型响应"""
        raise NotImplementedError("Subclass must implement fast_run method")

    def run(self, prompt: str, **kwargs) -> str:
        """运行模型，支持额外参数"""
        return self.fast_run(prompt)


class ZHIPULLM(LLM):
    """智谱AI的LLM接口（完整修改版）"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.failed_count = 0
        self.max_failures = 10

    def fast_run(self, prompt: str) -> str:
        """调用智谱AI API"""
        if self.failed_count >= self.max_failures:
            return self._get_smart_fallback_response(prompt)

        prompt = self._sanitize_prompt(prompt)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 0.9
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    self.failed_count = 0
                    return result['choices'][0]['message']['content'].strip()
                elif response.status_code == 400:
                    print(f"API请求格式错误: {response.text}")
                    return self._get_smart_fallback_response(prompt)
                elif response.status_code == 401:
                    print(f"API认证失败，请检查API密钥")
                    self.failed_count += 1
                    return self._get_smart_fallback_response(prompt)
                else:
                    print(f"API请求失败，状态码: {response.status_code}")

            except requests.exceptions.Timeout:
                print(f"API请求超时 (尝试 {attempt + 1}/{self.max_retries})")
            except Exception as e:
                print(f"API调用异常 (尝试 {attempt + 1}/{self.max_retries}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))

        self.failed_count += 1
        return self._get_smart_fallback_response(prompt)

    def _sanitize_prompt(self, prompt: str) -> str:
        """清理prompt中可能导致API错误的特殊字符"""
        prompt = prompt.replace('{', '').replace('}', '')
        prompt = re.sub(r'(.)\1{10,}', r'\1\1\1', prompt)
        if len(prompt) > 1000:
            prompt = prompt[:1000] + "..."
        return prompt

    def _get_smart_fallback_response(self, prompt: str) -> str:
        """提供更智能的后备响应"""
        if "请只输出一个数字" in prompt or "选择" in prompt:
            if "MainState" in prompt:
                if "愤怒" in prompt or "极度" in prompt:
                    return "1"
                elif "查看" in prompt:
                    return "3"
                else:
                    return str(random.choice([0, 1, 3, 5]))
            elif "CommentState" in prompt:
                if "愤怒" in prompt:
                    return "1"
                else:
                    return "2"
            elif "查看的评论编号" in prompt:
                return str(random.randint(0, 3))

        elif "评估" in prompt and "情绪" in prompt:
            base_anger = 0.3
            base_anxiety = 0.3
            base_disappointment = 0.3

            if "愤怒" in prompt or "生气" in prompt:
                base_anger += 0.3
            if "焦虑" in prompt or "担心" in prompt:
                base_anxiety += 0.3
            if "失望" in prompt or "沮丧" in prompt:
                base_disappointment += 0.3

            return json.dumps({
                "anger": min(1.0, base_anger + random.uniform(-0.1, 0.1)),
                "anxiety": min(1.0, base_anxiety + random.uniform(-0.1, 0.1)),
                "disappointment": min(1.0, base_disappointment + random.uniform(-0.1, 0.1))
            }, ensure_ascii=False)

        elif "发表" in prompt and "评论" in prompt:
            if "愤怒" in prompt or "负面情绪" in prompt:
                responses = [
                    "这种情况真是让人无法接受！为什么总是这样？",
                    "太让人失望了，这个社会到底怎么了？",
                    "我真的受够了！这种不公平何时才能结束？",
                    "看到这些我就生气，为什么没人管管？"
                ]
            elif "理性" in prompt or "引导" in prompt:
                responses = [
                    "大家冷静一下，我们可以理性讨论这个问题。",
                    "建议大家查证一下信息来源，不要被情绪左右。",
                    "这个问题确实值得关注，但我们需要更多事实依据。"
                ]
            elif "安抚" in prompt:
                responses = [
                    "我理解大家的心情，但愤怒解决不了问题。",
                    "确实让人担心，不过相信会有解决办法的。",
                    "大家的心情我都懂，让我们一起想想怎么改善。"
                ]
            else:
                responses = [
                    "这个问题值得深思。",
                    "说得有道理，我也是这么想的。",
                    "希望情况能够好转。",
                    "大家都不容易，互相理解吧。"
                ]
            return random.choice(responses)

        elif "更新" in prompt and "记忆" in prompt:
            return "我刚才看到了一些让人深思的内容，心情有些复杂。"

        elif "观点" in prompt and "变化" in prompt:
            return "经过思考，我觉得这个问题比想象的更复杂，需要多角度看待。"

        elif "社会信心" in prompt:
            current = 50
            match = re.search(r'当前社会信心[^\d]*(\d+)', prompt)
            if match:
                current = int(match.group(1))

            if "失望" in prompt or "愤怒" in prompt:
                change = random.randint(-10, -5)
            elif "希望" in prompt or "相信" in prompt:
                change = random.randint(0, 10)
            else:
                change = random.randint(-5, 5)

            return str(max(0, min(100, current + change)))

        elif "判断" in prompt and "恶意" in prompt:
            if any(word in prompt for word in ["攻击", "辱骂", "歧视", "威胁"]):
                return "1"
            else:
                return "0"

        else:
            return "我理解了。"

    def _get_fallback_response(self, prompt: str) -> str:
        """向后兼容的方法"""
        return self._get_smart_fallback_response(prompt)


class MockLLM(LLM):
    """用于测试的模拟LLM（完整修改版）"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.response_count = 0

    def fast_run(self, prompt: str) -> str:
        """生成模拟响应"""
        self.response_count += 1

        zhipu = ZHIPULLM(self.config)
        return zhipu._get_smart_fallback_response(prompt)


def create_LLM(config: Dict[str, Any]) -> LLM:
    """根据配置创建LLM实例"""
    model_name = config['LLM_config'].get('model_name', 'mock')

    if config['LLM_config'].get('use_mock', False):
        print("使用模拟LLM模式")
        return MockLLM(config['LLM_config'])

    if model_name.startswith('glm') or model_name.startswith('chatglm'):
        llm = ZHIPULLM(config['LLM_config'])
        if not config['LLM_config'].get('API_KEY'):
            print("警告：未提供API密钥，使用模拟LLM")
            return MockLLM(config['LLM_config'])
        return llm
    elif model_name == 'mock':
        print("使用模拟LLM进行测试")
        return MockLLM(config['LLM_config'])
    else:
        return ZHIPULLM(config['LLM_config'])