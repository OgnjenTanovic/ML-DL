from flask import Flask, request, render_template

app = Flask(__name__)


prediction = []
data = []


@app.route("/", methods=['GET', 'POST'])
def home():

    data = request.form['commit']
    print(data)
    return render_template('home.html', data=data, prediction=5)
    
    
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port='8000')
