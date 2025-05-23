import os
from typing import List
from copy import deepcopy
import warnings
from tqdm import tqdm
import numpy as np
import random
import time

import asyncio
from openai import AsyncOpenAI, AsyncAzureOpenAI

class OpenaiGenerator:
    """Class for api-based openai models"""

    def __init__(self, config):
        self.model_name = config["generator_model"]
        self.batch_size = config["generator_batch_size"]
        self.generation_params = config["generation_params"]
        
        # 通用模型配置
        # 用户可在配置文件设置自己需要的模型
        
        # 检测模型类型并选择对应的配置
        
        # OpenRouter模型检测
        
        # 检测模型类型并选择对应的配置
        if "gemini" in self.model_name.lower():
            self.openai_setting = config["google_setting"]
        elif "claude" in self.model_name.lower():
            self.openai_setting = config["anthropic_setting"]
        elif "deepseek" in self.model_name.lower():
            self.openai_setting = config["deepseek_setting"]
        # 其他模型类型判断...
        else:
            self.openai_setting = config["openai_setting"]
            
        # 统一处理所有模型的API密钥配置
        # 处理多API密钥配置（字符串或列表形式）
        if "api_keys" in self.openai_setting and self.openai_setting["api_keys"]:
            self.api_keys = self.openai_setting["api_keys"].split(",") if isinstance(self.openai_setting["api_keys"], str) else self.openai_setting["api_keys"]
            # 移除空字符串或空白项
            self.api_keys = [key.strip() for key in self.api_keys if key.strip()]
            if self.api_keys:
                # 创建独立的随机数生成器
                self.api_key_random = random.Random(time.time())
                # 为保持兼容性，设置第一个密钥为当前api_key
                self.openai_setting["api_key"] = self.api_keys[0]
                # 创建一个client字典以存储每个密钥对应的client
                self.clients = {}
            else:
                self.api_keys = None
                self.clients = None
        else:
            # 单个API密钥情况
            self.api_keys = None
            self.clients = None
        # print(self.model_name)
        # print(self.openai_setting)
        if self.openai_setting.get("api_key") is None:
            self.openai_setting["api_key"] = os.getenv("OPENAI_API_KEY")

        # 添加 base_url 支持
        if "api_base" in self.openai_setting:
            self.base_url = self.openai_setting["api_base"]
        else:
            self.base_url = None

        # 处理单个client或多个client的情况
        client_settings = deepcopy(self.openai_setting)
        
        # 设置较长的超时时间，避免大型请求超时
        client_settings["timeout"] = 60.0  # 设置为60秒
        
        # 从客户端设置中移除非客户端初始化参数
        if "api_keys" in client_settings:
            del client_settings["api_keys"]
        
        if "api_type" in client_settings and client_settings["api_type"] == "azure":
            del client_settings["api_type"]
            # 确保 base_url 被正确传递给 AsyncAzureOpenAI
            if self.base_url:
                client_settings["base_url"] = self.base_url
            
            # 创建客户端 - 对所有模型类型适用相同的逻辑
            if self.api_keys:
                # 为每个API密钥创建一个client
                for api_key in self.api_keys:
                    key_settings = deepcopy(client_settings)
                    key_settings["api_key"] = api_key
                    self.clients[api_key] = AsyncAzureOpenAI(**key_settings)
                
                # 同时保留单个client以保持兼容性
                self.client = self.clients[self.openai_setting["api_key"]]
            else:
                # 常规单一client初始化
                self.client = AsyncAzureOpenAI(**client_settings)
        else:
            # 确保 base_url 被正确传递给 AsyncOpenAI
            if self.base_url:
                client_settings["base_url"] = self.base_url
            
            # 创建客户端 - 对所有模型类型适用相同的逻辑
            if self.api_keys:
                # 为每个API密钥创建一个client
                for api_key in self.api_keys:
                    key_settings = deepcopy(client_settings)
                    key_settings["api_key"] = api_key
                    self.clients[api_key] = AsyncOpenAI(**key_settings)
                
                # 同时保留单个client以保持兼容性
                self.client = self.clients[self.openai_setting["api_key"]]
            else:
                # 常规单一client初始化
                self.client = AsyncOpenAI(**client_settings)

        # 移除 tiktoken 相关代码，因为实际上并未使用 tokenizer

    def _get_next_client(self):
        """获取API client，实现随机选择机制"""
        # 如果没有多个API密钥可用，则使用默认client
        if not self.api_keys:
            return self.client
        
        # 对所有模型类型适用相同的随机选择逻辑
        # 使用独立的随机数生成器随机选择一个API密钥
        random_index = self.api_key_random.randint(0, len(self.api_keys) - 1)
        current_key = self.api_keys[random_index]
        return self.clients[current_key]
    
    async def get_response(self, input: List, **params):
        # 所有模型均使用统一的轮询机制
        retries = 3  # 最大重试次数
        last_error = None
        
        for attempt in range(retries):
            current_key = None
            try:
                # 判断是否可以使用多密钥轮询
                if self.api_keys:
                    # 每次尝试都获取新的客户端，确保API密钥轮换
                    client = self._get_next_client()
                    # 获取当前密钥信息便于调试
                    for key, client_obj in self.clients.items():
                        if client_obj == client:
                            current_key = key
                            break
                    # 打印当前使用的密钥前十位字符
                    key_prefix = current_key[:10] + "..." if current_key else "unknown"
                    print(f"[尝试 {attempt+1}/{retries}] 使用API密钥: {key_prefix}")
                    
                    # 构建完整的请求参数
                    request_params = deepcopy(params)
                    
                    # 所有模型类型统一使用相同的API调用方式
                    response = await client.chat.completions.create(model=self.model_name, messages=input, **request_params)
                else:
                    # 单个密钥情况，使用默认client
                    response = await self.client.chat.completions.create(model=self.model_name, messages=input, **params)
                    
                return response.choices[0]
            except Exception as e:
                last_error = e
                # 判断是否是API限制错误，并且是否有多个API密钥可用
                if ("429" in str(e) or "rate" in str(e).lower() or "limit" in str(e).lower() or "quota" in str(e).lower()) and self.api_keys:
                    key_prefix = current_key[:10] + "..." if current_key else "unknown"
                    print(f"API密钥限制或配额耗尽: {key_prefix}, 切换到随机密钥 (尝试 {attempt+1}/{retries})")
                    print(f"错误详情: {str(e)}")
                    # 短暂等待，避免连续请求
                    await asyncio.sleep(1)
                    continue
                # 其他错误直接抛出
                raise e
        
        # 所有重试都失败
        print(f"[错误] 所有API密钥尝试失败")
        raise last_error

    async def get_batch_response(self, input_list: List[List], batch_size, **params):
        total_input = [self.get_response(input, **params) for input in input_list]
        all_result = []
        for idx in tqdm(range(0, len(input_list), batch_size), desc="Generation process: "):
            batch_input = total_input[idx : idx + batch_size]
            batch_result = await asyncio.gather(*batch_input)
            all_result.extend(batch_result)

        return all_result
        
    def _filter_thinking_chain(self, text):
        """过滤掉输出中的思维链部分
        
        Args:
            text (str): 原始响应文本
            
        Returns:
            str: 过滤后的文本
        """
        if not text:
            return text
            
        # 过滤 <think>...</think> 格式的思维链
        import re
        filtered_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        return filtered_text.strip()

    def generate(self, input_list: List[List], batch_size=None, return_scores=False, **params) -> List[str]:
        # deal with single input
        if len(input_list) == 1 and isinstance(input_list[0], dict):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        # deal with generation params
        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            generation_params.pop("do_sample")

        # 原来的max_tokens设置代码（已注释）
        # max_tokens = params.pop("max_tokens", None) or params.pop("max_new_tokens", None)
        # if max_tokens is not None:
        #     generation_params["max_tokens"] = max_tokens
        # else:
        #     generation_params["max_tokens"] = generation_params.get(
        #         "max_tokens", generation_params.pop("max_new_tokens", None)
        #     )
        
        # 写死max_tokens为10000，不再从参数或配置中读取
        generation_params["max_tokens"] = 8192
        # 清理可能存在的其他token限制参数
        params.pop("max_tokens", None)
        params.pop("max_new_tokens", None)
        generation_params.pop("max_new_tokens", None)

        if return_scores:
            if generation_params.get("logprobs") is not None:
                generation_params["logprobs"] = True
                warnings.warn("Set logprobs to True to get generation scores.")
            else:
                generation_params["logprobs"] = True

        if generation_params.get("n") is not None:
            generation_params["n"] = 1
            warnings.warn("Set n to 1. It can minimize costs.")
        else:
            generation_params["n"] = 1

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.get_batch_response(input_list, batch_size, **generation_params))
        
        # parse result into response text and logprob
        scores = []
        response_text = []
        for res in result:
            # 过滤思维链
            filtered_content = self._filter_thinking_chain(res.message.content)
            response_text.append(filtered_content)
            if return_scores:
                score = np.exp(list(map(lambda x: x.logprob, res.logprobs.content)))
                scores.append(score)
        if return_scores:
            return response_text, scores
        else:
            return response_text