import sys
sys.path.append('../2.MachineLearning_part')

from flask import Flask, request, render_template
from predictions import make_prediction

app = Flask(__name__)


prediction = []
data = []


@app.route("/", methods=['GET', 'POST'])
def home():
    global data
    global prediction
    
    if request.form:
        data = request.form['commit']        
        prediction = make_prediction(data)
    print(prediction)
    return render_template('home.html', data=data, prediction=prediction)
    
    
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port='8000')
