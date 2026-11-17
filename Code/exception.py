"""
异常处理模块：定义自定义异常类
"""


class TrendSimException(Exception):
    """TrendSim基础异常类"""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class LLMException(TrendSimException):
    """LLM相关异常"""
    
    def __init__(self, error_type: str, prompt: str = None, response: str = None):
        self.error_type = error_type
        self.prompt = prompt
        self.response = response
        
        message = f"LLM Error [{error_type}]"
        details = {}
        
        if prompt:
            details['prompt'] = prompt[:200] + '...' if len(prompt) > 200 else prompt
        if response:
            details['response'] = response[:200] + '...' if len(response) > 200 else response
            
        super().__init__(message, details)
        
    def __str__(self):
        base_msg = f"LLM异常 [{self.error_type}]"
        
        if self.error_type == 'Agent-Decision-Fail-Action-Code':
            return f"{base_msg}: 代理决策失败，无法解析动作代码。响应: {self.response[:100] if self.response else 'None'}"
        elif self.error_type == 'Agent-Decision-Fail':
            return f"{base_msg}: 代理决策失败，动作代码无效。"
        elif self.error_type == 'Detailed-Comment-Index-None-Int':
            return f"{base_msg}: 无法解析评论索引，期望整数但得到: {self.response}"
        elif self.error_type == 'API-Call-Failed':
            return f"{base_msg}: API调用失败。"
        else:
            return super().__str__()


class SimulationException(TrendSimException):
    """模拟相关异常"""
    
    def __init__(self, error_type: str, **kwargs):
        self.error_type = error_type
        
        messages = {
            'Invalid-State': "无效的状态转换",
            'Agent-Not-Found': "找不到指定的代理",
            'Time-System-Error': "时间系统错误",
            'Configuration-Error': "配置错误",
            'Data-Loading-Error': "数据加载错误"
        }
        
        message = messages.get(error_type, f"模拟错误: {error_type}")
        super().__init__(message, kwargs)


class DataException(TrendSimException):
    """数据相关异常"""
    
    def __init__(self, error_type: str, file_path: str = None, **kwargs):
        self.error_type = error_type
        self.file_path = file_path
        
        messages = {
            'File-Not-Found': f"找不到文件: {file_path}",
            'Invalid-Format': f"文件格式无效: {file_path}",
            'Parse-Error': f"解析文件失败: {file_path}",
            'Write-Error': f"写入文件失败: {file_path}"
        }
        
        message = messages.get(error_type, f"数据错误: {error_type}")
        details = {'file_path': file_path} if file_path else {}
        details.update(kwargs)
        
        super().__init__(message, details)


class EmotionException(TrendSimException):
    """情绪系统相关异常"""
    
    def __init__(self, error_type: str, agent_id: int = None, **kwargs):
        self.error_type = error_type
        self.agent_id = agent_id
        
        messages = {
            'Invalid-Emotion-Value': "无效的情绪值",
            'Emotion-Update-Failed': "情绪更新失败",
            'Emotion-Analysis-Failed': "情绪分析失败",
            'Contagion-Error': "情绪传染计算错误"
        }
        
        message = messages.get(error_type, f"情绪系统错误: {error_type}")
        details = {'agent_id': agent_id} if agent_id is not None else {}
        details.update(kwargs)
        
        super().__init__(message, details)


class InterventionException(TrendSimException):
    """干预系统相关异常"""
    
    def __init__(self, error_type: str, intervention_type: str = None, **kwargs):
        self.error_type = error_type
        self.intervention_type = intervention_type
        
        messages = {
            'Invalid-Intervention-Type': f"无效的干预类型: {intervention_type}",
            'Intervention-Failed': "干预执行失败",
            'Trigger-Error': "干预触发条件错误",
            'Agent-Creation-Failed': "干预代理创建失败"
        }
        
        message = messages.get(error_type, f"干预系统错误: {error_type}")
        details = {'intervention_type': intervention_type} if intervention_type else {}
        details.update(kwargs)
        
        super().__init__(message, details)


def handle_exception(func):
    """异常处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TrendSimException as e:
            print(f"[TrendSim异常] {e}")
            raise
        except Exception as e:
            print(f"[未预期的异常] {type(e).__name__}: {e}")
            raise TrendSimException(f"未预期的错误: {str(e)}")
    
    return wrapper