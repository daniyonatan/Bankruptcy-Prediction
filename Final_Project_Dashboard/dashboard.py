from flask import Flask, render_template, request
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

app = Flask(__name__)

df = pd.read_csv('data.csv')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/dataset/', methods = ('POST','GET'))
def dataset():
    return render_template('dataset.html', tables = [df.to_html(
        classes = 'data', header = 'true', justify = 'center',
        max_rows = 500, max_cols = 15)])


if __name__ == "__main__":
    app.run(debug = True)