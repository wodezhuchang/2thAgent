# -*- coding: utf-8 -*-
"""专家端 chat 验收测试。"""
import requests

BASE = "http://localhost:8000"
results = []

def test(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))
    print(f"[{status}] {name}")
    if detail:
        print(f"       {detail[:120]}")

print("=== 专家端 chat 验收 ===")

# 1. 基本对话
r = requests.post(f"{BASE}/api/expert-chat", json={
    "expert_id": "prof_zhang",
    "message": "这个学生之前试讲过几次？进步趋势怎么样？",
    "student_id": "demo_user"
})
data = r.json()
test("专家端对话200", r.status_code == 200)
test("返回reply", bool(data.get("reply")), f"reply长度: {len(data.get('reply', ''))}")
test("意图识别为student_history", data.get("intent") == "student_history", f"intent: {data.get('intent')}")
session_id = data.get("session_id")

# 2. 课标查询
r = requests.post(f"{BASE}/api/expert-chat", json={
    "expert_id": "prof_zhang",
    "message": "初中数学一次函数的课标要求是什么？",
    "session_id": session_id
})
data = r.json()
test("课标查询意图正确", data.get("intent") == "curriculum_ref", f"intent: {data.get('intent')}")
test("课标查询有回复", bool(data.get("reply")))

# 3. 评语辅助
r = requests.post(f"{BASE}/api/expert-chat", json={
    "expert_id": "prof_zhang",
    "message": "这个学生导入时间过长，板书布局混乱，口头禅多。帮我整理一段评语",
    "session_id": session_id
})
data = r.json()
test("评语辅助意图正确", data.get("intent") == "comment_draft", f"intent: {data.get('intent')}")
test("评语辅助有回复", bool(data.get("reply")))

# 4. 分歧分析
r = requests.post(f"{BASE}/api/expert-chat", json={
    "expert_id": "prof_zhang",
    "message": "机器分39.6，我打了45，差5.4分，哪里可能有问题？",
    "session_id": session_id
})
data = r.json()
test("分歧分析意图正确", data.get("intent") == "disagreement_analysis", f"intent: {data.get('intent')}")

# 5. 对话入库
import sqlite3
conn = sqlite3.connect("data/memory/agent_memory.db")
count = conn.execute("SELECT COUNT(*) FROM chat_message WHERE user_id='prof_zhang'").fetchone()[0]
test("专家对话已入库", count >= 4, f"共{count}条")
conn.close()

# 汇总
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
