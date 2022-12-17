from flask import Flask, render_template, request
from models.alexnet import model_run
from utils import class_img

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_word(): return render_template('index.html')
    
@app.route('/', methods=['POST'])
def predict():
    
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    name, precentage = class_img(model_run(), image_path)
    
    classification = '%s (%.2f%%)' % (name, precentage)

    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)