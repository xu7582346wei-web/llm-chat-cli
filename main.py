import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

resp = client.chat.completions.create(
    #model="glm-4-flash",
    model="qwen2.5:7b",
    messages=[
        {"role": "user", "content": "用 Python 写一个冒泡排序"}
    ],
)

print(resp.choices[0].message.content)
print("---- usage ----")
print(resp.usage)
print("model:", resp.model)
print("finish_reason:", resp.choices[0].finish_reason)
