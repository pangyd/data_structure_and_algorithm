import base64
from io import BytesIO
from PIL import Image
import cv2
import sys
import requests
import json
import time


def cv2_image_to_base64(image_path):
    image = cv2.imread(image_path)
    # 将图片数据保存为字节对象
    is_success, buffer = cv2.imencode(".jpg", image)
    if not is_success:
        raise Exception("Could not encode image to JPEG format.")
    # 将字节对象转换为Base64字符串
    base64_str = base64.b64encode(buffer).decode('utf-8')

    return base64_str


# 读取图片
imgpath = sys.argv[1]
base64_str = cv2_image_to_base64(imgpath)
# 发送请求
data = {'image': base64_str}
url = 'http://10.211.18.139:8860/dewarp_predict'
response = requests.post(url, json=data)
outimg = response.json()['dewarp_img']
# 保存图片
base64_bytes = base64.b64decode(outimg)
bytes_io = BytesIO(base64_bytes)
image = Image.open(bytes_io)
image.save("tmp_flask.jpg")