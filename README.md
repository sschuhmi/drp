# drp


## Project "Disaster Response Pipelines"

This project is included within Udacity´s 'Data Scientist' Nanodegree Program ([nd025](https://www.udacity.com/enrollment/nd025)).

### Table of Contents
 
1. [Project Motivation](#motivation)
2. [Survey Data](#surveydata)
3. [Provided Files](#files)
4. [Installation](#installation)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation <a name="motivation"></a>

This project includes a data set containing real messages that were sent during disaster events. 

Based on these, a machine learning pipeline was created to categorize these events so that one can send the messages to an appropriate disaster relief agency.

The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

## Project Data <a name="project_data"></a>

tbd

## Provided Files <a name="files"></a>

The following files are provided within this project:
<ul>
  <li><b>Jupyter Notebooks:</b> The notebooks for preparing the [ETL-Pipeline](https://github.com/sschuhmi/drp/blob/main/Jupyter%20Notebooks/ETL%20Pipeline%20Preparation.ipynb) and the ML-Pipeline are included in the corresponding subfolder.</li>
  <li><b>tbd:</b> tbd</li>
  <li><b>README.md:</b> This file</li>
</ul>

## Installation & Libraries <a name="installation"></a>

In order the install and run the disaster response app, the following steps have to be performed in a terminal session (please adapt filepaths and filenames where needed):
<ol>
  <li>Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
  </li>
  <li>Go to `app` directory: `cd app`</li>
  <li>Run your web app: `python run.py`</li>
  <li>Click the `PREVIEW` button to open the homepage</li>
</ol>

The required (Python) libraries are automatically installed within the execution of the Python scripts.

## Results <a name="results"></a>

The results are shown in a web app where the user can additionally enter messages to be classified by the established model.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Uses the message and category filenames provided by Udacity within its Data Science Nanodegree (nd025).
These have been provided by Udacity´s partner [Appen](https://www.figure-eight.com/) (formally: Figure 8).
