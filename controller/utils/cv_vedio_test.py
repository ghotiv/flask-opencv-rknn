import cv2
from rknnlite.api import RKNNLite
from rknn_image import process_image,letterbox
import PIL.Image as Image
import io

RKNN_MODEL = 'yolov5s.rknn'

rknn_lite = RKNNLite()

# load RKNN model
print('--> Load RKNN model')
ret = rknn_lite.load_rknn(RKNN_MODEL)
if ret != 0:
    print('Load RKNN model failed')
    exit(ret)
# Init runtime environment
print('--> Init runtime environment')
ret = rknn_lite.init_runtime()
if ret != 0:
    print('Init runtime environment failed!')
    exit(ret)

#获得视频的格式
fcap = cv2.VideoCapture('test.mp4')
  
#获得码率及尺寸
fps = fcap.get(cv2.CAP_PROP_FPS)
print(fps)
# size = (int(fcap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
#         int(fcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# fNUMS = fcap.get(cv2.CAP_PROP_FRAME_COUNT)

fcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
fcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

ret, frame = fcap.read()
print(ret)

if ret:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = letterbox(image)[0]
    outputs = rknn_lite.inference(inputs=[image])
    frame = process_image(image, outputs)
    print(frame)

    if frame is not None:
        ret, image = cv2.imencode('.jpg', frame)
        ibytes = image.tobytes()
        image_obj = Image.open(io.BytesIO(ibytes))
        image_obj.save('cv.jpg')
        print(ibytes)

fcap.release()
rknn_lite.release()