from flask import Flask, render_template, request
from models import *
from utils import class_img, run_model

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_word(): return render_template('index.html')
    
@app.route('/', methods=['POST'])
def predict():
    
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    model, name = run_model(GoogleNET())
    class_name, precentage = class_img(model, image_path)
    
    classification = '%s (%.2f%%) %s' % (class_name, precentage, name)

    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)