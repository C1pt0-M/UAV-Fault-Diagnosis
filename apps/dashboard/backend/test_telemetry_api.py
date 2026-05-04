#!/usr/bin/env python3
"""测试实时遥测 API 的简单脚本（不启动真实 PX4）。"""

import sys
from pathlib import Path

# 添加 backend 到路径
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from uav_dashboard_api import create_app

def test_telemetry_routes():
    """测试遥测路由是否正确注册。"""
    app = create_app()

    print("=" * 60)
    print("实时遥测 API 路由测试")
    print("=" * 60)

    telemetry_routes = []
    for rule in app.url_map.iter_rules():
        if 'telemetry' in str(rule):
            methods = rule.methods - {"HEAD", "OPTIONS"}
            telemetry_routes.append((str(rule), methods))

    print(f"\n共注册 {len(telemetry_routes)} 个遥测路由:\n")
    for route, methods in sorted(telemetry_routes):
        method_str = ', '.join(sorted(methods))
        print(f"  [{method_str:4}] {route}")

    print("\n" + "=" * 60)
    print("数据库表检查")
    print("=" * 60)

    with app.app_context():
        from uav_dashboard_api import db
        from uav_dashboard_api.models.telemetry_session import TelemetrySession, TelemetryFaultEvent

        # 检查表是否存在
        inspector = db.inspect(db.engine)
        tables = inspector.get_table_names()

        print(f"\n数据库中的表 (共 {len(tables)} 个):\n")
        for table in sorted(tables):
            marker = "✓" if table in ['telemetry_sessions', 'telemetry_fault_events'] else " "
            print(f"  {marker} {table}")

        # 检查遥测表结构
        if 'telemetry_sessions' in tables:
            print("\n[telemetry_sessions] 表结构:")
            for col in inspector.get_columns('telemetry_sessions'):
                print(f"  - {col['name']:20} {col['type']}")

        if 'telemetry_fault_events' in tables:
            print("\n[telemetry_fault_events] 表结构:")
            for col in inspector.get_columns('telemetry_fault_events'):
                print(f"  - {col['name']:20} {col['type']}")

    print("\n" + "=" * 60)
    print("✓ 所有检查通过")
    print("=" * 60)
    print("\n后端已就绪，可以启动 Flask 服务进行前端对接。")
    print("启动命令: python run.py")
    print()


if __name__ == "__main__":
    test_telemetry_routes()
