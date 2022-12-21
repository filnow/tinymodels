from flask import Flask, render_template, request
from models import *
from utils import class_img, run_model

app = Flask(__name__, template_folder='./frontend/templates',static_folder='./frontend/static')
@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')
@app.route('/', methods=['POST'])
def predict():
    
    model_name = request.form.get('myInput', False)
    
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    
    class_name, precentage = class_img(run_model(model_name), image_path)
    
    classification = f'Image is {class_name} in ({precentage:.2f}%) predicted by model: {model_name}'
    
    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)