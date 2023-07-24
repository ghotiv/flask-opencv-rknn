# from flask_script import Manager
from controller import create_app

# 创建APP对象,使用参数dev时开启debug
# app = create_app('dev')
app = create_app('pro')
# # 创建脚本管理
# mgr = Manager(app)


if __name__ == '__main__':
    app.run(threaded=True, host="0.0.0.0", port=5000)

