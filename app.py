from flask import Flask, render_template, request
from models import *
from utils import class_img, run_model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    
    model_name = request.form.get('myInput', False)
    imagefile = request.files['imagefile']
    
    image_path = "./images/" + imagefile.filename
    
    imagefile.save(image_path)
    model = run_model(model_name)
    class_name, precentage = class_img(model, image_path)
    
    classification = '%s (%.2f%%) %s' % (class_name, precentage, model_name)
    
    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)