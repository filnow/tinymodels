from flask import Flask, render_template, request, redirect, url_for
from utils import class_img, run_model


app = Flask(__name__, template_folder='./templates',static_folder='./static')

@app.route('/', methods=['GET', 'POST'])
def index():
    
    if request.method == 'POST':

        model_name = request.form.get('myInput', False)
        imagefile = request.files['imagefile']
        
        img_path = "./static/images/" + imagefile.filename
        imagefile.save(img_path)
        
        class_name, precentage = class_img(run_model(model_name), img_path)
        
        prediction = f'Image is {class_name} in ({precentage:.2f}%) predicted by model: {model_name}'
    
        return redirect(url_for('success', prediction=prediction, img_path=img_path))
    
    return render_template('index.html')

@app.route('/success')
def success():

    prediction = request.args.get('prediction')

    img_path = request.args.get('img_path')
    
    return render_template('success.html', prediction=prediction, img_path=img_path)


if __name__ == '__main__':
    app.run(port=3000, debug=True)