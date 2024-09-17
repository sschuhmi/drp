# drp


## Project "Disaster Response Pipelines"

This project is included within Udacity´s 'Data Scientist' Nanodegree Program ([nd025](https://www.udacity.com/enrollment/nd025)).

### Table of Contents
 
1. [Project Motivation](#motivation)
2. [Project Data](#project_data)
3. [Provided Files](#files)
4. [Installation](#installation)
5. [Results](#results)
6. [Sources & Licensing](#licensing)

## Project Motivation <a name="motivation"></a>

This project includes a data set containing real messages that were sent during disaster events. 

Based on these, a machine learning pipeline was created to categorize these events so that one can send the messages to an appropriate disaster relief agency.

The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

## Project Data <a name="project_data"></a>

The project feature a data set containing real messages that were sent during disaster events are were provided by Appen (formerly: Figure 8) for the Udacity projects.

Precisely, the following data is used:
- Over 25k Messages sent during disaster events
- Categorization of these messages according to 36 disaster message categories (e.g., "request", "offer", "medical_help", "fire", "storm")

## Provided Files <a name="files"></a>

The following folders and files are provided within this project:
<ul>
   <li><b>Data:</b> The input data for the project (message and categories) as well as the created SQLite database (drp.db) and the Python script for processing the data
   <li><b>Jupyter Notebooks:</b> The notebooks for preparing the ETL-Pipeline and the ML-Pipeline as well as some logging information from the data preparation process are included in the corresponding subfolder.</li>
  <li><b>app:</b> The Python script to run the web app which shows the results of the classification and additionally allows the user to enter further messages for classification, as well as the html files in the "templates" subfolder. An exemplary screenshot of the app with the visualizations is also given here.</li>
  <li><b>models:</b> The stored classifier in a Pickle file (classifier.pkl) and the Python script to train the classifier.</li>
  <li><b>README.md:</b> This help file to get started.</li>
  <li><b>drp.zip:</b> zip-Archive including folders "Data", "app" and "models" with included files.</li>
</ul>

## Installation & Libraries <a name="installation"></a>

In order the install and run the disaster response app, the following steps have to be performed in a terminal session (please adapt filepaths and filenames where needed):
<ol>
   <li>Unzip the file drp.zip in your Python directory to ensure the whole project is available</li>
   <li>Run the following commands in the project's root directory to set up your database and model. Hint: Please make sure you have the latest version of Pandas installed, to avoid incompatibility issues with SQLAchemy.
    <ul>
      <li>To run ETL pipeline that cleans data and stores in database:<br><b>python data/process_data.py data/messages.csv data/categories.csv data/drp.db</b></li>
      <li>To run ML pipeline that builds model, trains and evaluates classifier:<br><b>python models/train_classifier.py data/drp.db models/classifier.pkl</b></li>
    </ul>
  </li>
  <li>Go to 'app' directory: <b>cd app</b></li>
  <li>Run your web app: <b>python run.py</b></li>
  <li>Click the 'PREVIEW' button to open the homepage</li>
</ol>

Within this project, Python v3.6.3 was used. The required additional libraries are automatically installed within the execution of the Python scripts and feature some common Python libaries like NumPy or Pandas as well as models and metrics from scikit-learn.

## Results <a name="results"></a>

The results are shown in a web app where the user can additionally enter messages to be classified by the established model.

Besides the possibility to enter a message for classification (and see the classification results after pushing the button 'Classify message'), the following 5 visualizations are shown:
<ul>
 <li>Distribution of all messages according to genres ('news', 'direct', 'social')</li>
 <li>Distribution of all messages according to the 36 categories</li>
 <li>Distribution only of 'news' messages according to the 36 categories</li>
 <li>Distribution only of 'direct' messages according to the 36 categories</li>
 <li>Distribution only of 'social' messages according to the 36 categories</li>
</ul>

Here is an exemplary screenshot of the web app.

<img src="https://github.com/sschuhmi/drp/blob/main/app/AppScreen.jpeg" alt="Web app" width=window.screen.width>

## Sources & Licensing<a name="licensing"></a>

This projects relies on the messages and category filenames provided by Udacity within its Data Science Nanodegree (nd025).
These have been collected and prepared by Udacity´s partner [Appen](https://www.figure-eight.com/) (formerly: Figure 8) - please refer to them for license information.
