from flask import Flask, render_template, request, Response, send_from_directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

app = Flask(__name__)

df = pd.read_csv('data.csv')

def make_graph():
    feature = ['class','Attr1','Attr42','Attr19','Attr35','Attr21']
    labels = ['Below One','Above One']

    for i in feature:
        if i == 'class':
            plt.figure(figsize = (7,7))
            rects1 = sns.countplot(data = df, x = 'class')
            plt.title(i)
            plt.tight_layout()
            plt.savefig('static\images\{}.png'.format(i))
        elif i != 'class':
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

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result', methods = ['POST','GET'])
def result():
    if request.method == 'POST':
        input = request.form
        res = {
            'Attr27': float(input['POA']) / float(input['Fin_Ex']),
            'Attr21': float(input['Current_Sales']) / float(input['Prior_Sales']),
            'Attr26': (float(input['Net_Profit']) + float(input['Depreciation'])) / float(input['Current_Sales']),
            'Attr46': (float(input['Current_As']) - float(input['Inventory']) - float(input['Receivables'])) / float(input['Short_Lib']),
            'Attr6': float(input['Ret_Earn']) / float(input['Total_As']),
            'Attr38': float(input['Constant_Cap']) / float(input['Total_As']),
            'Attr39': float(input['POS']) / float(input['Current_Sales']),
            'Attr34': float(input['Op_Ex']) / float(input['Total_Lib']),
            'Attr5': ((float(input['Cash']) + float(input['Securities']) + float(input['Receivables']) - float(input['Short_Lib'])) / (float(input['Op_Ex']) - float(input['Depreciation']))) * 365,
            'Attr41': float(input['Total_Lib']) / ((float(input['POA']) + float(input['Depreciation'])) * (12/365)),
            'Attr58': float(input['Total_Cost']) / float(input['Current_Sales']),
            'Attr44': (float(input['Receivables']) * 365) / float(input['Current_Sales']),
            'Attr56': (float(input['Current_Sales']) - float(input['COGS'])) / float(input['Current_Sales']),
            'Attr24': float(input['GP3']) / float(input['Total_As']),
            'Attr42': float(input['POA']) / float(input['Current_Sales'])
        }
        x_test = pd.DataFrame(data = res, index = [0])
        predict = model.predict(x_test)
        return render_template('result.html', data = input, pred = ['Bankrupt' if predict[0] == 1 else 'Not Bankrupt'])

if __name__ == "__main__":
    model = joblib.load('final_model')
    app.run(debug = True)