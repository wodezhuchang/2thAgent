# -*- coding: utf-8 -*-
"""第二阶段验收测试脚本。"""
import requests
import json

BASE = "http://localhost:8000"
results = []

def test(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))
    print(f"[{status}] {name}")
    if detail:
        print(f"       {detail[:100]}")

# ========== 1. 待复评列表 ==========
print("=== 1. 待复评列表 ===")
r = requests.get(f"{BASE}/api/triage/pending", params={"limit": 10})
data = r.json()
test("待复评列表返回200", r.status_code == 200)
test("待复评列表有数据", data.get("count", 0) > 0, f"共{data.get('count')}条")

# 检查最新几条有没有 triage_route
trials = data.get("trials", [])
has_route = any(t.get("triage_route") for t in trials)
test("新记录有triage_route", has_route, f"最新: {trials[0] if trials else '无'}")

# ========== 2. 专家提交复评 ==========
print("\n=== 2. 专家提交复评 ===")
if trials:
    target_try = trials[0]["try_id"]
    review_body = {
        "try_id": target_try,
        "expert_id": "prof_zhang",
        "final_scores": {
            "D1": {"total": 42},
            "D2": {"total": 48},
            "D3": {"total": 38},
            "D4": {"total": 52},
            "overall_percent": 45.0
        },
        "comments": "导入环节时间偏长，建议压缩到1分钟内。板书布局可以左右分栏，左边写定义右边写例题。教态自然，但口头禅'然后'偏多，注意控制。"
    }
    r = requests.post(f"{BASE}/api/expert-review", json=review_body)
    data = r.json()
    test("专家复评提交200", r.status_code == 200)
    test("返回机器分vs专家分差异", "delta" in data, f"machine={data.get('machine_overall')}, expert={data.get('expert_overall')}, delta={data.get('delta')}")
    test("有分歧维度", "disagree_dims" in data, f"分歧维度: {data.get('disagree_dims')}")

# ========== 3. 消息系统 ==========
print("\n=== 3. 消息系统 ===")
r = requests.get(f"{BASE}/api/messages", params={"user_id": "demo_user"})
data = r.json()
test("拉未读消息200", r.status_code == 200)
test("有专家复评消息", data.get("count", 0) > 0, f"共{data.get('count')}条未读")
msgs = data.get("messages", [])
if msgs:
    test("消息类型正确", msgs[0].get("type") == "expert_review_ready")
    test("消息关联了试讲", bool(msgs[0].get("related_try_id")))

# 标记已读
if msgs:
    read_body = {"user_id": "demo_user", "message_ids": [msgs[0]["message_id"]]}
    r = requests.post(f"{BASE}/api/messages/read", json=read_body)
    data = r.json()
    test("标记已读成功", data.get("marked") >= 1)

# 再拉应该少了
r = requests.get(f"{BASE}/api/messages", params={"user_id": "demo_user"})
data = r.json()
test("已读后未读数减少", data.get("count", 0) < len(msgs), f"现在{data.get('count')}条")

# ========== 4. chat 聊天助手 ==========
print("\n=== 4. chat 聊天助手 ===")

# 4.1 基本对话
r = requests.post(f"{BASE}/api/chat", json={"user_id": "demo_user", "message": "我上次试讲多少分？"})
data = r.json()
test("chat基本对话200", r.status_code == 200)
test("返回了reply", bool(data.get("reply")), f"reply长度: {len(data.get('reply', ''))}")
test("有意图识别", bool(data.get("intent")), f"intent: {data.get('intent')}")
session_id = data.get("session_id")

# 4.2 怎么练类问题
r = requests.post(f"{BASE}/api/chat", json={"user_id": "demo_user", "message": "板书总是写得很乱，怎么练？", "session_id": session_id})
data = r.json()
test("怎么练类问题意图正确", data.get("intent") == "how_to_practice", f"intent: {data.get('intent')}")
test("怎么练类有回复", bool(data.get("reply")))

# 4.3 情绪状态类
r = requests.post(f"{BASE}/api/chat", json={"user_id": "demo_user", "message": "我觉得练不好了，好焦虑", "session_id": session_id})
data = r.json()
test("情绪类意图正确", data.get("intent") == "state_check", f"intent: {data.get('intent')}")
test("情绪类有回复", bool(data.get("reply")))

# 4.4 陪练类
r = requests.post(f"{BASE}/api/chat", json={"user_id": "demo_user", "message": "你是评委，问我一个问题吧", "session_id": session_id})
data = r.json()
test("陪练类意图正确", data.get("intent") == "practice_drill", f"intent: {data.get('intent')}")

# 4.5 对话历史入库
r = requests.get(f"{BASE}/api/messages", params={"user_id": "demo_user"})  # 顺便确认
# 直接查数据库验证 chat_message 表
import sqlite3
conn = sqlite3.connect("data/memory/agent_memory.db")
count = conn.execute("SELECT COUNT(*) FROM chat_message WHERE user_id='demo_user'").fetchone()[0]
test("chat消息已入库", count >= 4, f"共{count}条chat记录")
conn.close()

# ========== 汇总 ==========
print("\n=== 验收汇总 ===")
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
total = len(results)
print(f"总计: {total}项, 通过: {passed}, 失败: {failed}")
if failed:
    print("\n失败项:")
    for name, status, detail in results:
        if status == "FAIL":
            print(f"  - {name}: {detail}")
