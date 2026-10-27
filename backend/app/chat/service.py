# -*- coding: utf-8 -*-
"""试讲陪练聊天助手。

定位：一个"精通试讲的朋友"角色，不是通用聊天机器人。
- 查自己：历史分数、成长趋势、训练计划
- 怎么练：针对性练习建议、教案讨论、临场问题应对
- 陪练：模拟评委提问、时间规划
- 状态：情绪支持
"""
from __future__ import annotations
import re
import uuid
from datetime import datetime

from app.llm import llm_client
from app.memory import memory_store
from app.config import settings


# 角色设定：朋友式，不说"作为AI助手"，说人话
CHAT_SYSTEM_PROMPT = """【角色】
你是杨德琰的试讲陪练朋友，不是冷冰冰的AI客服。你自己也是师范生长大的，知道上台紧张、
板书歪、口头禅多是什么感受。你说话像一个靠谱的学长/师姐：先共情，再给具体动作，
不空洞鼓励，不说教。

【说话风格】
1. 不说"作为一个AI助手"、"建议你可以考虑"、"根据我的分析"
2. 说人话，有具体动作："你上次导入讲了3分钟还没进正题，下次试试抛一个问题，
   1分钟引出课题"
3. 先共情再给建议："紧张正常，我第一次上台手都在抖。你试试..."
4. 短句为主，不要长篇大论
5. 不知道就说不知道，不瞎编

【你的能力】
- 他的历史试讲分数、各维度表现、进步趋势
- 他当前的短板和训练计划
- 怎么练试讲：板书、教态、语言、教学设计
- 临场问题怎么应对：紧张、学生答不上来、时间不够
- 模拟评委提问陪练

【边界】
- 只聊试讲相关的，不聊无关的
- 他问的问题超纲了就说"这个我帮不上，你得问老师"
- 不替他写完整教案，只给方向和建议
"""


def detect_intent(message: str) -> str:
    """轻量意图识别。"""
    msg = message.lower()

    # 查自己：历史分数、进步、趋势
    if re.search(r"(我上次|最近|进步|分数|得分|成绩|多少分|之前|历史|成长|趋势|哪个维度|短板|弱项)", msg):
        return "self_progress"

    # 怎么练：练习建议、教案、方法
    if re.search(r"(怎么练|怎么办|如何改|怎么改|练习方法|板书怎么|教态怎么|语言怎么|口头禅|紧张怎么办|教案|导入|小结|提问)", msg):
        return "how_to_practice"

    # 陪练：模拟评委、提问、排计划
    if re.search(r"(模拟|陪练|评委|问我|排个计划|练习计划|下周|比赛|时间规划)", msg):
        return "practice_drill"

    # 状态：情绪、心态
    if re.search(r"(我觉得|练不好|崩溃|焦虑|害怕|紧张|没信心|不想练|累了|太难了)", msg):
        return "state_check"

    return "casual"


def build_user_profile(user_id: str) -> str:
    """从 memory_store 读用户画像：最近3次分数、短板、训练计划。"""
    lines = []

    # 最近3次试讲
    try:
        history = memory_store.get_user_trials(user_id, limit=3)
        if history.get("trials"):
            lines.append("【最近试讲记录】")
            for i, t in enumerate(history["trials"]):
                lines.append(f"  第{i+1}次：总分 {t.get('overall_percent', '?')}，时间 {t.get('created_at', '')[:10]}")
    except Exception:
        pass

    # 成长曲线
    try:
        growth = memory_store.get_growth_timeline(user_id)
        pts = growth.get("points", [])
        if len(pts) >= 2:
            latest = pts[-1]
            prev = pts[-2]
            delta = round(latest["overall"] - prev["overall"], 1)
            arrow = "↑进步" if delta > 0 else ("↓退步" if delta < 0 else "→持平")
            lines.append(f"【成长趋势】最近一次 {latest['overall']} 分，比上次 {arrow} {abs(delta)} 分")
            # 找最弱维度
            dims = latest.get("dims", {})
            if dims:
                weak_dim = min(dims.items(), key=lambda x: x[1])
                dim_names = growth.get("dim_names", {})
                weak_name = dim_names.get(weak_dim[0], weak_dim[0])
                lines.append(f"  当前最弱：{weak_name}（{weak_dim[1]}分）")
    except Exception:
        pass

    return "\n".join(lines) if lines else "（暂无历史记录）"


def chat(user_id: str, message: str, session_id: str | None = None) -> dict:
    """主入口：处理一次 chat 请求。"""
    if not session_id:
        session_id = f"chat_{uuid.uuid4().hex[:8]}"

    intent = detect_intent(message)

    # 读用户画像
    profile = build_user_profile(user_id)

    # 读最近对话历史
    history = memory_store.get_chat_history(user_id, session_id, limit=10)
    history_text = "\n".join([f"{'我' if h['role']=='user' else '你'}：{h['content']}" for h in history])

    # 根据意图决定要不要查 RAG
    rag_context = ""
    if intent in ("how_to_practice", "practice_drill"):
        try:
            from app.rag import retriever
            hits = retriever.retrieve(message, top_k=3)
            if hits:
                rag_context = retriever.build_context(hits, max_chars=1500)
        except Exception:
            rag_context = ""

    # 组装 prompt
    system = CHAT_SYSTEM_PROMPT + f"\n\n【他的情况】\n{profile}"
    if rag_context:
        system += f"\n\n【相关教学参考】\n{rag_context}"
    system += f"\n\n【最近对话】\n{history_text}"

    # 调 LLM
    client = llm_client.get_llm_client()
    try:
        resp = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        reply = resp.choices[0].message.content.strip()
    except Exception as e:
        reply = f"（系统开小差了：{e}）你再问一次？"

    # 存对话
    memory_store.save_chat_message(user_id, "user", message, intent, session_id)
    memory_store.save_chat_message(user_id, "assistant", reply, intent, session_id)

    return {
        "reply": reply,
        "intent": intent,
        "session_id": session_id,
    }
