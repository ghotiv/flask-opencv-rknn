from flask import session, render_template, request, redirect, url_for, Response, jsonify
from controller.modules.home import home_blu
from controller.utils.camera import VideoCamera
#import time
#import cv2

video_camera = None
global_frame = None

# 主页
@home_blu.route('/')
def index():
    # 模板渲染
    username = session.get("username")
    if not username:
        return redirect(url_for("user.login"))
    return render_template("index.html")

# 获取视频流
def video_stream():
    global video_camera
    global global_frame

    if video_camera is None:
        video_camera = VideoCamera()

    while True:
        #start_time = time.time()
        frame = video_camera.get_frame()
        #end_time = time.time()
        #print('get_frame cost %f second' % (end_time - start_time))
        #time.sleep(0.01)
        if frame is not None:
            global_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')


# 视频流
@home_blu.route('/video_viewer')
def video_viewer():
    # 模板渲染
    username = session.get("username")
    if not username:
        return redirect(url_for("user.login"))
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# 状态
@home_blu.route('/record_status', methods=['POST'])
def record_status():
    global video_camera
    if video_camera is None:
        video_camera = VideoCamera()

    json = request.get_json()

    status = json['status']

    if status == "true":
        video_camera.start_record()
        return jsonify(result="started")
    else:
        video_camera.stop_record()
        return jsonify(result="stopped")


# 识别状态
@home_blu.route('/process_status', methods=['POST'])
def process_status():
    global video_camera
    if video_camera is None:
        video_camera = VideoCamera()

    json = request.get_json()

    process_status = json["status"]

    if process_status == "true":
        video_camera.start_process()
        return jsonify(result="process")
    else:
        video_camera.stop_process()
        return jsonify(result="pause")
