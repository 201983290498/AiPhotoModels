import json

from flask import Flask, request, jsonify
from cifar10 import cifar10
import base64
# import cifar100 as cf100
app = Flask(__name__)
model1 = cifar10.mymodel()
# model2 = cf100.mymodel('my_train_model1.hdf5')
@app.route('/imgCfy', methods=['POST'])
def func():
    if request.method == 'POST':
        imagebase = request.get_data().decode("utf-8")
        imagebase = json.loads(imagebase)
        imageurl = base64.b64decode(imagebase.get("b64"))
        with open('temp.png', 'wb') as file:
            file.write(imageurl)
        # categy=cf100.predict_img('temp.png',model2)
        categy=cifar10.test('temp.png')
        print(categy)
        return categy


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8899)