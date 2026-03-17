import requests
import os
from io import BytesIO
from PIL import Image
import json
from time import sleep


def request_chain(json: dict, timeout=60):
    langchain_server_url = "http://190.92.231.77:880/iwencai/dialog/chain/execute"
    langchain_headers = {
        "Host": "aime-langchain-engine-server",
        "X-Arsenal-Auth": "aime-reinforcement-learning-environment-access",
        "Content-Type": "application/json",
    }

    response = requests.post(langchain_server_url, headers=langchain_headers, json=json, timeout=timeout)
    return response


def parse_search_data(input, data):
    res_list = []
    for i in data:
        res = ''
        res += f"搜索问句: {input}\n"
        res += f"标题: {i['title']}\n"
        if i.get('publish_time'):
            res += f"时间: {i['publish_time']}\n"
        if i.get('summary'):
            res += f"摘要: {i['summary']}\n"
        if i.get('full_summary'):
            res += f"内容: {i['full_summary']}\n"
        if i.get('url'):
            res += f"站点:\n溯源地址：{i['url']}\n"
        res_list.append(res)

    return res_list


def parse_info_content_data(input, data):
    res_list = []
    for i in data:
        res = ''
        res += f"标题: {i['title']}\n"
        if i.get('publish_time'):
            res += f"时间: {i['publish_time']}\n"
        if i.get('summary'):
            res += f"摘要: {i['summary']}\n"
        if i.get('full_summary'):
            res += f"内容: {i['full_summary']}\n"
        if i.get('url'):
            res += f"站点:\n溯源地址：{i['url']}\n"
        res_list.append(res)

    return res_list


def uploaded_image_message(img_path, img_url=None) -> list[dict]: # 如果是图，就返回一个列表
    img_url = img_url if img_url else 'Unknown'
    return[[
        {
            "type": "text",
            'text': f'<image_data>\n<image_url>{img_url}</image_url>'
        },
        {
            "type": "image_url",
            "image_url": {
                "url": img_path,
            },
        },
        {
            "type": "text",
            'text': f'</image_data>'
        }
    ]]


def obtaininfosummary(input, theme_id="TZ-10605"):
    obj = {
    "chain_name": "ObtainInfoSummary",
    "req_type": "nostream",
    "user_id": "zhangjian",
    "session_id": "a59e8949afa345a12ced2ce4044c5c11",
    "question_id": "c47fa3e9-9bb0-4458-a302-add195d92711",
    "trace_id": "debug_platform_bb0d3928a36d43c4afeac69ba327abb1",
    "debug": False,
    "source": "ths_mobile_yuyinzhushou",
    "human_message": "{'pageId':'1'}",
    "question": "帮我在算力 板块挑选一个优秀的股票，要求政策利好",
    "model_param": {},
    "history": [],
    "add_info": {"theme_id": "TZ-11939"}, #TZ-11939
    "request_id": "ef535d5e-85b3-4c46-88de-19c32ad3006a",
    "version": "v1",
    "client_id": ""
    }
    obj["human_message"] = input
    obj["add_info"]["theme_id"] = theme_id

    req = request_chain(obj)
    resp = req.json()
    response = resp.get("response", {})
    if not response:
        return ['工具调用失败']

    results = response.get("result", [])
    if not results:
        return ['工具调用失败']
    
    result_data = results[0]
    parse_data = result_data['text']
    return [parse_data]


def obtaininfocontent(input, theme_id="TZ-11939"):
    obj = {
    "chain_name": "ObtainInfoContent",
    "req_type": "nostream",
    "user_id": "zhangjian",
    "session_id": "a59e8949afa345a12ced2ce4044c5c11",
    "question_id": "c47fa3e9-9bb0-4458-a302-add195d92711",
    "trace_id": "debug_platform_bb0d3928a36d43c4afeac69ba327abb1",
    "debug": False,
    "source": "ths_mobile_yuyinzhushou",
    "human_message": "{'pageId':'1','numbers':'2,3,6,10'}",
    "question": "帮我在算力 板块挑选一个优秀的股票，要求政策利好",
    "model_param": {},
    "history": [],
    "add_info": {"theme_id": "TZ-11939"},
    "events": [
        {
            "event_type": "user_input",
            "event_name": "deep_research",
            "content": {}
        }
    ],
    "request_id": "ef535d5e-85b3-4c46-88de-19c32ad3006a",
    "version": "v1",
    "client_id": ""
    }

    obj["human_message"] = input
    obj["add_info"]["theme_id"] = theme_id

    req = request_chain(obj)
    resp = req.json()
    response = resp.get("response", {})
    if not response:
        return [f'Search工具，输入{input}，调用失败。']

    results = response.get("result", [])
    if not results:
        return [f'Search工具，输入{input}，调用失败。']
    
    result_data = results[0]
    raw_data: list[dict[str, str]] = result_data.get("raw_data", [])
    parse_data = parse_info_content_data(input, raw_data) if raw_data else [f'Search工具，输入{input}，调用失败。']
    return parse_data



def search(input):
    obj = {"chain_name": "Search", "req_type": "nostream", "events":[{"event_name":"deep_research","event_type":"user_input"}], "human_message": input}

    req = request_chain(obj)
    resp = req.json()
    response = resp.get("response", {})
    if not response:
        return [f'Search工具，输入{input}，调用失败。']

    results = response.get("result", [])
    if not results:
        return [f'Search工具，输入{input}，调用失败。']
    
    result_data = results[0]
    raw_data: list[dict[str, str]] = result_data.get("raw_data", [])
    parse_data = parse_search_data(input, raw_data) if raw_data else [f'Search工具，输入{input}，调用失败。']
    return parse_data


def finquery(input):
    obj = {"chain_name": "FinQuery", "req_type": "nostream", "human_message": input}

    req = request_chain(obj)
    resp = req.json()

    response = resp.get("response", {})
    if not response:
        return [f'FinQuery工具，输入{input}，调用失败。']

    results = response.get("result", [])
    if not results:
        return [f'FinQuery工具，输入{input}，调用失败。']

    result_data = results[0]
    
    parse_data = result_data['text']
    return [f'取数问句: {input}\n 取数结果: {parse_data}']


def image_message(img_path, img_url=None): # 如果是图，就返回一个列表
    img_url = img_url if img_url else 'Unknown'
    return[[
        {
            "type": "text",
            'text': f'<image_data>\n<image_url>{img_url}</image_url>'
        },
        {
            "type": "image_url",
            "image_url": {
                "url": img_path,
            },
        },
        {
            "type": "text",
            'text': f'\n</image_data>'
        }
    ]]


def ticker_chart(input, images_dir):
    input = json.loads(input)
    query = {
        "startDate": input.get("startDate"),
        "endDate": input.get("endDate"),
        "codeName": input.get("codeName"),
        "chartType": input.get("chartType"),
        "indicator": input.get("indicator")
    }
    query = json.dumps(query, ensure_ascii=False, separators=(",", ":"))
    req_json = r"""{
    "chain_name": "TickerChart",
    "req_type": "nostream",
    "user_id": "125",
    "session_id": "143",
    "question_id": "143",
    "trace_id": "1746001144320",
    "debug": false,
    "source": "aicubes_agent_77",
    "human_message": "{\"startDate\":\"2023-01-01\",\"chartType\":\"Weekly Candlestick\",\"endDate\":\"2025-04-30\",\"codeName\":\"同花顺\",\"indicator\":[\"MA\",\"MACD\",\"RSI\",\"BOLL\"]}",
    "question": "{\"startDate\":\"2023-01-01\",\"chartType\":\"Weekly Candlestick\",\"endDate\":\"2025-04-30\",\"codeName\":\"MSFT\",\"indicator\":[\"MA\",\"MACD\",\"RSI\",\"BOLL\"]}",
    "stream": false}"""
    req = json.loads(req_json)
    req["human_message"] = query
    req["question"] = query
    resp = request_chain(req)
    if not resp.json()['response']:
        return [f'TickerChart工具，输入{input}，调用失败。']
    resp = resp.json()['response']['result'][0]
    if "media_info" not in resp:
        return [f'TickerChart工具，输入{input}，调用失败。']
    url = resp["media_info"]["url"]

    if url is None:
        return [f'TickerChart工具，输入{input}，调用失败。']
    url_response = requests.get(url)
    image_data = url_response.content
    image = Image.open(BytesIO(image_data))
    filename = url.split("/")[-1]
    # 构建完整的文件路径
    filepath = os.path.join(images_dir, filename)
    # 将图像数据保存到文件
    image = image.convert('RGB')
    image.save(filepath)
    
    return [{"image_path": filepath, "image_url": url}] # 返回的是一个列表，里面是字典


def find_twin_chart(input):
    input = json.loads(input)
    image_url = input['url']
    req_json = {
            "chain_name": "ChartTwinFinder",
            "req_type": "nostream",
            "user_id": "wangqihan",
            "session_id": "554bae8dad8183ad57abddc231ce0186",
            "question_id": "67024549-97cd-419b-aeaf-a17580d265d8",
            "trace_id": "debug_platform_f55697ed761e4f77ac1004460fcd3116",
            "debug": False,
            "source": "ths_mobile_yuyinzhushou",
            "human_message": "",
            "question": "",
            "add_info": {
                "input_type": "typewrite",
                "task_type": "offline_batch_data",
                "rela_trace_ids": [],
                "device_type": "pc",
                "question_risk_tags": [],
                "user_lang": "",
                "urp_data_permission": "hideChargeData",
                "urp_data_permission_bit": "",
                "product_data": [],
                "component_version": "1.1.3",
                "merge_repeat": True,
                "ability_version": "basic",
                "txt_to_image_processing_num": 0,
                "txt_to_image_task_id": "",
                "txt_to_image_seed": "",
                "image_to_txt_download_url": "http://oss.10jqka.com.cn/iwc-web-userinfo-storage-server.model-image-q-a/bcf0d0684dc6432793e8de8436140b6d.png",
                "stock_code": "",
                "frontend_version": "3.4.1",
                "fallback": False
                },
                "action_param": None,
                "result_page_info": None,
                "chain_vanish_request": None,
                "user_name": None,
                "client_ip": None,
                "action_name": "图像生文本",
                "thought_infos": [],
                "accept_content": None,
                "events": [
                    {
                        "event_type": "user_input",
                        "event_name": "get_image_summary",
                        "content": {
                            "name": "screenshots.png",
                            "public_url": "https://ai.iwencai.com/userinfo-model-image-q-a/bcf0d0684dc6432793e8de8436140b6d.png",
                            "download_url": "http://oss.10jqka.com.cn/iwc-web-userinfo-storage-server.model-image-q-a/bcf0d0684dc6432793e8de8436140b6d.png"
                        }
                    },
                    {
                        "event_type": "user_input",
                        "event_name": "deep_thinking",
                        "content": {}
                    }
                ],
                "agent_config": None,
                "agent_id": "6a597186b3e4468cac90eee29673bb5e",
                "transfer_question": "",
                "request_id": "c2f07e6a-281b-4db2-9cd9-ea8ed6a759f8",
                "agent_chain_request": None,
                "messages": None,
                "multimodal_messages": None,
                "version": "v1",
                "client_id": "",
                "stream": False
        }
    req = req_json
    req['add_info']['image_to_txt_download_url'] = image_url
    req['events'][0]['content']['download_url'] = image_url
    resp = request_chain(req)
    if not resp.json()['response']:
        return [f'ChartTwinFinder工具，输入{input}，调用失败。']
    resp = resp.json()['response']['result'][0]
    
    return [resp['text']]


def visit_web(input):
    req_json = r"""{
    "chain_name": "VisitWeb",
    "req_type": "nostream",
    "user_id": "125",
    "session_id": "143",
    "question_id": "143",
    "trace_id": "1746001144320",
    "debug": false,
    "source": "aicubes_agent_77",
    "human_message": "",
    "question": "",
    "stream": false}"""
    req = json.loads(req_json)
    req["human_message"] = input
    req["question"] = input
    resp = request_chain(req)
    if not resp.json()['response']:
        return [f'VisitWeb工具，输入{input}，调用失败。']
    resp = resp.json()['response']['result'][0]
    
    return [resp['text']]


tool_map = {
    "FinQuery": finquery,
    "Search": search,
    "TickerChart": ticker_chart,
    "ChartTwinFinder": find_twin_chart,
    "VisitWeb": visit_web,
    "ObtainInfoSummary": obtaininfosummary,
    "ObtainInfoContent": obtaininfocontent
}


def get_tools_results(tools, images_dir=r'/mnt/HithinkOmniSSD/user_workspace/ganziliang/code/agent/check_images'):
    if images_dir and not os.path.exists(images_dir):
        os.makedirs(images_dir)
    tools_results = []
    prev_tool = None
    
    for tool in tools:
        tool_name = tool['name']
        tool_input = tool['input']
        if prev_tool and tool_name == prev_tool and tool_name == 'Search':
            sleep(1)
        try:
            if tool_name == 'ObtainInfoSummary' or tool_name == 'ObtainInfoContent':
                result = tool_map[tool_name](tool_input, tool['theme_id'])
            elif tool_name == 'TickerChart':
                result = tool_map[tool_name](tool_input, images_dir)
            else:
                result = tool_map[tool_name](tool_input)
            tools_results.extend(result)
        except Exception as e:
            print(f"Error running tool {tool_name}: {e}")
        prev_tool = tool_name
        
    return tools_results # finquery 和 search 返回的是 list，内容是字符串。图片相关工具返回的是 list[list]。


def check_tool_calls(tools, images_dir=r'/mnt/HithinkOmniSSD/user_workspace/ganziliang/code/agent/check_images'):
    if images_dir and not os.path.exists(images_dir):
        os.makedirs(images_dir)
    checked_results = []

    prev_tool = None
    
    for tool in tools:
        tool_name = tool['name']
        tool_input = tool['input']
        if prev_tool and tool_name == prev_tool and tool_name == 'Search':
            sleep(1)
        try:
            result = tool_map[tool_name](tool_input) if tool_name != 'TickerChart' else tool_map[tool_name](tool_input, images_dir)
            if tool_name == 'FinQuery': # FinQuery好像没找到数据，也会返回找到0条数据
                if not result or '取数结果：为您找到0条数据' in result[0]:
                    checked_results.append(False)
                else:
                    checked_results.append(True)
            else:
                checked_results.append(True) if result else checked_results.append(False)
        except Exception as e:
            checked_results.append(False)
            print(f"Error running tool {tool_name}: {e}")
        prev_tool = tool_name
        
    return checked_results # finquery 和 search 返回的是 list，内容是字符串。图片相关工具返回的是 list[list]。


if __name__ == '__main__':
    # obtaininfocontent('{"pageId":"1","numbers":"2,3,6,10"}', theme_id="TZ-986")
    # obtaininfosummary("{'pageId':'1'}", theme_id="TZ-986")
    res = get_tools_results([
                    # {"name": "ChartTwinFinder", "input": '{\"query\": \"分析与下图形态走势相近的股票\", \"url\": \"http://oss.myhexin.com/iwc-web-userinfo-storage-server.model-image-q-a/2d801a7ae3d6450ca8104be2390d54bb.jpg\"}'},
                            #  {"name": "ChartTwinFinder", "input": '{\"query\": \"分析与下图形态走势相近的股票\", \"url\": \"./dataset1/Images/001201_20250415_20250617.png\"}'},
                    # {"name": "ChartTwinFinder", "input": '{\"query\": \"分析与下图形态走势相近的股票\", \"url\": \"./dataset1/Images/001201_20250415_20250617.png\"}'},
                    # {"name":"Search", "input": "马云"},
                    # {"name":"Search", "input": "德生科技 信息安全解决方案"},
                    # {"name":"Search", "input": "ST信通(600289) 深度研究报告"},
                    # {'name': 'TickerChart', 'input': '{"codeName": "300584", "chartType": "Daily Candlestick", "startDate": "2025-05-19", "endDate": "2025-06-11", "indicator": ["MA"]}'}
                    {'name': 'FinQuery', 'input': '茅台的股票代码·'}
                    # {"name": "VisitWeb", "input": "https://news.10jqka.com.cn/20251030/c672126342.shtml"},
                    # {"name":"FinQuery", "input": "上一周黄金价格如何"},
                    # {'name': 'TickerChart', 'input': '{"codeName": "同花顺", "chartType": "Daily Candlestick", "startDate": "2024-11-05", "endDate": "2025-05-05", "indicator": ["MA", "MACD"]}'}
                    # {"name":"TickerChart", "input": '{\"codeName\": \"688585.SH\", \"chartType\": \"Daily Candlestick\", \"startDate\": \"2025-04-09\", \"endDate\": \"2025-07-09\", \"indicator\": [\"MA\", \"VOL\"]}'}
                    # {\"name\": \"TickerChart\", \"input\": \"{\\\"codeName\\\": \\\"云创数据\\\", \\\"chartType\\\": \\\"Daily Candlestick\\\", \\\"startDate\\\": \\\"2025-06-17\\\", \\\"endDate\\\": \\\"2025-07-07\\\", \\\"indicator\\\": [\\\"MA\\\", \\\"MACD\\\", \\\"RSI\\\"]}\"}
                    ])
    # res = check_tool_calls([
        # {'name': 'TickerChart', 'input': '{"codeName": "605333", "chartType": "Daily Candlestick", "startDate": "2024-11-05", "endDate": "2025-05-05", "indicator": ["MA", "MACD"]}'},
        # {'name': 'TickerChart', 'input': '{"codeName": "605333", "chartType": "Weekly Candlestick", "startDate": "2024-11-05", "endDate": "2025-05-05", "indicator": ["MA", "MACD"]}'},
        # {'name': 'TickerChart', 'input': '{"codeName": "601825", "chartType": "Weekly Candlestick", "startDate": "2024-11-05", "endDate": "2025-05-05", "indicator": ["MA", "MACD"]}'}
        # {"name":"Search", "input": "欧元区最新季调后就业人数年率数据 2025年第一季度"},
        # {"name":"Search", "input": "地缘政治紧张局势对欧元区就业影响"},
        # {"name":"Search", "input": "欧元区2025年就业增长预测 欧洲央行"},
        # {"name":"Search", "input": "关税争端对欧元区就业影响"},
        # {"name":"FinQuery", "input": "欧元区主要国家季调后就业人数年率数据"},
        # {"name":"FinQuery", "input": "欧元区季调后就业人数年率 2019年至今"},
        # {"name":"FinQuery", "input": "欧洲央行欧元区季调后就业人数年率数据"},
        # {"name":"FinQuery", "input": "广西柳工集团有限公司的股票代码"},
    # ])
    print(res)