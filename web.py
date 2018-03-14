import cv2
import numpy as np
from flask import Flask
from flask import request

from imagenet_classes import classes
from keras_model import KerasManager

app = Flask(__name__)
manager = KerasManager()
manager.start()
keras_model = manager.KerasModel()
keras_model.initialize()


@app.route('/hello', methods=['POST'])
def hello():
    img = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), -1)
    img = cv2.resize(img, (224, 224))
    return classes[np.argmax(keras_model.predict(img))]


if __name__ == '__main__':
    app.run()
