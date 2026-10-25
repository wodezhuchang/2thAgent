# -*- coding: utf-8 -*-
import requests, json, sys
url = "http://127.0.0.1:8000/api/evaluate_video"
video = r"D:\2thAgent\datasets\demo_preteaching\video\mypreteaching_fore.mp4"
with open(video, "rb") as f:
    files = {"video": ("mypreteaching_fore.mp4", f, "video/mp4")}
    data = {"user_id":"demo_user","subject":"数学","lesson":"一次函数",
            "lesson_type":"新授课","grade_level":"初中","duration_seconds":"180"}
    print("上传并评价中（约1-2分钟）...")
    r = requests.post(url, files=files, data=data, timeout=300)
print("status:", r.status_code)
if r.status_code != 200:
    print(r.text[:500]); sys.exit()
j = r.json()
print("overall:", j.get("scores",{}).get("overall_percent"))
print("sensing_steps:")
for s in j.get("sensing_steps",[]):
    print(f"  {s['tool']}: ok={s['ok']} {s.get('error','')}")
print("elapsed:", j.get("sensing_elapsed"))
print("trial_id:", j.get("trial_id"))
print("weaknesses:", j.get("weaknesses"))
