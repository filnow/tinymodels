from flask import Flask, render_template, request, redirect, url_for
from models import *
from utils import class_img, run_model

app = Flask(__name__, template_folder='./frontend/templates',static_folder='./frontend/static')

@app.route('/', methods=['GET', 'POST'])
def index():
    
    if request.method == 'POST':

        model_name = request.form.get('myInput', False)
        
        imagefile = request.files['imagefile']
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)
        
        class_name, precentage = class_img(run_model(model_name), image_path)
        
        prediction = f'Image is {class_name} in ({precentage:.2f}%) predicted by model: {model_name}'
    
        return redirect(url_for('success', prediction=prediction))
    
    return render_template('index.html')

@app.route('/success')
def success():

    prediction = request.args.get('prediction')
    return render_template('success.html', prediction=prediction)


if __name__ == '__main__':
    app.run(port=3000, debug=True)