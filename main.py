# from flask import Flask, render_template, request, redirect
# import torch
# from mnist_model import MNISTModel  
# import threading

# app = Flask(__name__)

# # Define a list to store ongoing and finished experiments
# ongoing_experiments = []
# finished_experiments = []

# @app.route('/')
# def index():
#     return render_template('index.html', ongoing=ongoing_experiments, finished=finished_experiments)

# @app.route('/run_experiments', methods=['POST'])
# def run_experiments():
#     # Parse hyperparameters from the form data
#     learning_rate = float(request.form['learning_rate'])
#     batch_size = int(request.form['batch_size'])
#     epochs = int(request.form['epochs'])

#     model = MNISTModel()

#     # Train the model with the specified hyperparameters
#     experiment_thread = threading.Thread(target=train_model, args=(model, learning_rate, batch_size, epochs))
#     experiment_thread.start()

#     return redirect('/')

# def train_model(model, learning_rate, batch_size, epochs):
#     progress = model.train(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs)
#     ongoing_experiments.append({'learning_rate': learning_rate, 'batch_size': batch_size, 'epochs': epochs, 'progress': progress})

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect
from mnist_model import MNISTModel  

app = Flask(__name__)

# Define a list to store ongoing experiments
ongoing_experiments = []

@app.route('/')
def index():
    return render_template('index.html', ongoing=ongoing_experiments)

import threading

@app.route('/run_experiments', methods=['POST'])
def run_experiments():
    # Parse hyperparameters from the form data
    learning_rate = float(request.form['learning_rate'])
    batch_size = int(request.form['batch_size'])
    epochs = int(request.form['epochs'])

    model = MNISTModel()

    # Train the model with the specified hyperparameters and get the progress
    progress = train_model(model, learning_rate, batch_size, epochs)

    # Add the experiment details and progress to the ongoing experiments list
    ongoing_experiments.append({
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'progress': progress
    })

    return redirect('/')

def train_model(model, learning_rate, batch_size, epochs):
    progress = model.train(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs)
    return progress

if __name__ == '__main__':
    app.run(debug=True)







