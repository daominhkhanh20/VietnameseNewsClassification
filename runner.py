from flask import Flask, request
from flask.templating import render_template
import sys
from model import Vocabulary,LabelSmoothingCrossEntropyLoss
import warnings
from pipeline import get_label
if not sys.warnoptions:
    warnings.simplefilter("ignore")


app=Flask(__name__)
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/", methods=['GET', 'POST'])
def form_post():
    text=request.form['inputtext']
    label,model_name=get_label(text)
    return render_template("submit.html",model_name=model_name, result=label)

if __name__ == '__main__':
    app.run(debug=True)

#sudo netstat -tulnp | grep :5000
#sudo kill -9 pid
