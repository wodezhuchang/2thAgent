# -*- coding: utf-8 -*-
"""预置演示/测试账号（幂等，可重复执行）。

对应《多角色改造计划书_v1.0.md》第 8 节「sandbox 虚拟账号来源」——采用 seed 脚本预置，
便于演示与测试；admin 亦可通过 /api/users 建号。

账号（真实 real）：
- admin / admin123   全知全能（系统管理员）
- teacher / 123456   李老师（教师）
- demo / 123456      师范生小张（学生，已有历史评价）
- student2 / 123456  师范生小王（学生）

账号（沙箱 sandbox，开发者测试用）：
- s_demo / 123456    虚拟师范生
- s_teacher / 123456 虚拟教师

师生指导关系（评价分发依据）：
- teacher_1 → demo_user, student_2
- sandbox_teacher → sandbox_student
"""
from app.memory import memory_store


def main() -> None:
    memory_store.init_db()
    # 真实账号
    memory_store.create_user("admin", "admin", "admin123", name="系统管理员",
                             role="admin", account_type="real", org="华中师范大学")
    memory_store.create_user("teacher_1", "teacher", "123456", name="李老师",
                             role="teacher", account_type="real", org="华中师范大学")
    memory_store.create_user("demo_user", "demo", "123456", name="师范生小张",
                             role="student", account_type="real", org="华中师范大学")
    memory_store.create_user("student_2", "student2", "123456", name="师范生小王",
                             role="student", account_type="real", org="华中师范大学")
    # sandbox 虚拟账号（开发者测试）
    memory_store.create_user("sandbox_student", "s_demo", "123456", name="虚拟师范生",
                             role="student", account_type="sandbox", org="测试沙箱")
    memory_store.create_user("sandbox_teacher", "s_teacher", "123456", name="虚拟教师",
                             role="teacher", account_type="sandbox", org="测试沙箱")
    # 师生指导关系
    memory_store.add_teacher_student("teacher_1", "demo_user")
    memory_store.add_teacher_student("teacher_1", "student_2")
    memory_store.add_teacher_student("sandbox_teacher", "sandbox_student")
    print("SEED_OK users:", len(memory_store.list_users()))


if __name__ == "__main__":
    main()
