from __future__ import annotations

import os
import time

from uav_dashboard_api import create_app


def main() -> None:
    app = create_app()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    debug = bool(app.config.get("DEBUG", False))

    print("=" * 50)
    print("基于数据驱动的无人机故障智能诊断系统 - Dashboard Backend")
    print("=" * 50)
    print(f"启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"服务地址: http://{host}:{port}")
    print(f"API 地址: http://{host}:{port}/api")
    print(f"健康检查: http://{host}:{port}/health")
    print(f"配置环境: {app.config.get('APP_ENV', 'unknown')}")
    print("=" * 50)
    print("后端服务启动中...")
    print()

    try:
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    finally:
        from uav_dashboard_api.services.telemetry_manager import shutdown_telemetry_manager

        shutdown_telemetry_manager(app)


if __name__ == "__main__":
    main()
