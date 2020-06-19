from flask import Flask, render_template, request, Response, send_from_directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

df = pd.read_csv('data.csv')

def make_graph():
    feature = ['Attr1','Attr42','Attr19','Attr35','Attr21']
    labels = ['Below One','Above One']

    for i in feature:
        not_bankrupt = [sum(df[df['class'] == 0][i] < 1), sum(df[df['class'] == 0][i] >= 1)]
        bankrupt = [sum(df[df['class'] == 1][i] < 1), sum(df[df['class'] == 1][i] >= 1)]
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize = (7,7))
        rects1 = ax.bar(x - width/2, not_bankrupt, width, label = 'Not-Bankrupt')
        rects2 = ax.bar(x + width/2, bankrupt, width, label = 'Bankrupt')

        ax.set_ylabel('Count')
        ax.set_title(i)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')


        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()
        plt.savefig('static\images\{}.png'.format(i))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dataset/', methods = ('POST','GET'))
def dataset():
    return render_template('dataset.html', tables = [df.to_html(
        classes = 'data', header = 'true', justify = 'center', max_rows = 10)])

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory('./static/images', filename)

@app.route('/viz/')
def viz():
    make_graph()
    graph_names = os.listdir("./static/images")
    print(graph_names)
    return render_template('viz.html', graph_names = graph_names)


if __name__ == "__main__":
    app.run(debug = True)