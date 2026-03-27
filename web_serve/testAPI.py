import requests
import json

if __name__ == '__main__':
    url = "https://api.modelarts-maas.com/v2/chat/completions"  # API地址
    api_key = "yGwxmOi_Howrdi_ayyRyfjA6LbUp2jULKTgsGYUMC1EsaPhNZ1VtWWkJLYJhmKk6NB9IuDdDAiUHhxSUMl3mlQ"  # 把MAAS_API_KEY替换成已获取的API Key

    # Send request.
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        "model": "deepseek-v3.2",  # model参数，您可按需更换模型参数
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "介绍下你自己"}
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)

    # Print result.
    print(response.status_code)
    print(response.text)