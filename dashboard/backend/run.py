# 应用运行入口文件
import os
import time

from app import create_app

app = create_app()


if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5000'))
    debug = os.getenv('FLASK_DEBUG', 'True') == 'True'

    print('=' * 50)
    print('无人机故障诊断系统 - Dashboard Backend')
    print('=' * 50)
    print(f'启动时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'服务地址: http://{host}:{port}')
    print(f'API地址: http://{host}:{port}/api')
    print(f'健康检查: http://{host}:{port}/health')
    print('=' * 50)
    print('后端服务启动中...')
    print()

    app.run(host=host, port=port, debug=debug, use_reloader=False)
