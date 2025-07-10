import pandas as pd
import plotly.graph_objs as go
from flask import Flask, render_template, request

app = Flask(__name__)

def create_plot(x, y, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=title))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title=title)
    return fig.to_html(full_html=False)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == 'admin' and password == 'admin12345':
            df = pd.read_csv('sensor_data.csv')
            df1 = df.dropna()
            x = df1['time']

            temp_plot = create_plot(x, df1['SensorB'], 'Temperature')
            pressure_plot = create_plot(x, df1['SensorA'], 'Pressure')
            humidity_plot = create_plot(x, df1['SensorC'], 'Humidity')

            return render_template(
                'second.html',
                username=username,
                temp_plot=temp_plot,
                pressure_plot=pressure_plot,
                humidity_plot=humidity_plot
            )
        else:
            return render_template('error.html', username=username)
    
    return render_template('main_entery.html')

if __name__ == '__main__':
    app.run(debug=True)
