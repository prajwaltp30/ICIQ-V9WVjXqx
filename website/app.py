from flask import Flask, render_template, request, redirect, url_for
import subprocess
import os
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_script():
    if request.method == 'POST':
        subprocess.Popen(['python', 'merge.py'])
        return 'Script started successfully'

    return render_template('index.html')

@app.route('/results', methods=['GET', 'POST'])  # Allow both GET and POST requests
def show_results():
    if request.method == 'POST':
        # Handle form submission if needed
        pass



    # Determine if an additional graph exists
    additional_graph_exists = os.path.exists("final_graph.png")

    return render_template('results.html', additional_graph_exists=additional_graph_exists)

    # Read BPM data from the file
    bpm_data = []
    with open("bpm_values.txt", "r") as file:
        for line in file:
            bpm = float(line.strip())
            bpm_data.append(bpm)

    # Calculate average BPM between 5 to 20 seconds
    average_bpm = np.mean(bpm_data[5:])

    return render_template('index.html', final_readings=final_readings, additional_graph_exists=additional_graph_exists, average_bpm=average_bpm)

if __name__ == '__main__':
    app.run(debug=True)
