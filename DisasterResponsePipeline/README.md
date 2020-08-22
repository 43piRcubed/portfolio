# Project: Disaster Response Pipeline

## Content:

- [Overview](#overview)
- [Project Elements](#elements)
  - [ETL Pipeline](#etl_)
  - [ML Pipeline](#ml_)
  - [Flask App](#flask)
- [Execution](#execute)
  - [Data Cleaning](#cleaning)
  - [Classifier Training](#training)
  - [Starting the Web App](#starting)
- [Conclusion](#conclusion)
- [Files](#files)
- [Requirements](#req)
- [Credits](#credits)

***

<a id='overview'></a>

## 1.  Overview

In this project disaster response data from <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a> is analyzed to build a model for an API that allows for the classification of disaster messages based on 36 different categories.

The aim is to categorize messages received to facilitate the notification to the appropriate disaster relief agencies .

On the back end the data is processed and modeled.  The front end is a web app that allows an emergency operator to enter a message, which will then be categorized accordingly to allow for the appropriate actions to be taken.

The main dashboard of the app also includes visualizations that summarize the data used as modelling input. 

<a id='elements'></a>

## 2.  Project Components

There are three components of this project:
-  an ETL pipeline
-  a Machine Learning pipeline
-  a flask app

<a id='etl_'></a>

### 2.1  ETL Pipeline

The purpose of the ETL pipeline is:

- Load the {messages} and {categories} datasets
- Merge the datasets
- Clean the data
- Store the data in an **SQLite database**: _data/DisasterResponse.db_

The ETL pipleine can be found in **_data/process_data.py_**

<a id='ml_'></a>

### 2.2  ML Pipeline

The purpose of the ML pipeline is:

- Load data from the **SQLite database**
- Split the data into training and testing sets
- Train and tune a model using GridSearchCV
- Output the results on the test set
- Export the final model to a pickle file

the ML pipeline can be found in **_models/train_classifier.py_**

<a id='flask'></a>

### 2.3  Flask App

The Flask web app's main dashboard provides the main user interface as well as overview visualizations of the dataset.
The main user interface allows emergency operators to enter a meesage they received, i.e. _"Napa flooded as Sacramento River burst over shores. Rescue help needed"_.
The message will be classified and assigned categories out of 36 available. This allows the emrgency operators to take the next steps efficently and effectively.

Here are some screen shots of the web app:

**_Screenshot 1  -  Main Dashboard_**

![dashboard](media/MainDashboard.png)

What the app will do is that it will classify the text message into categories so that appropriate relief agency can be reached out for help.

**_Screenshot 2  -  Response page_**

![result1](media/result1.png)
![result2](media/result2.png)

<a id='execute'></a>

## 3. Executing

Three steps need to be taken to get up and running:

- Clean the data
- Run the classifier
- Launch the web app

<a id='cleaning'></a>

### 3.1 Data Cleaning

Requirements:

- path and name of ETL pipeline code file
- path and name of messages dataset file
- path and name of categories dataset file
- path and name of output database file

**Go to the main project directory** and the run the command in this format:

_python path/'ETL pipeline' path/'messeages dataset file' path/'categories dataset file' path/'database file'_

this turns into the following command with its 3 arguments:

```bat
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
 
This will run the ETL pipeline and create a database file with the specified name.  If the database file already exists it will be replaced with the new one.

below a screen shot of the process:

_**Screenshot 4  -  Processing the Data**_

![process_data](media/running_ETL.png)

<a id='training'></a>

### 3.2 Classifier Training

After the data cleaning process, run this command **from the project directory**:

```bat
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

This will use cleaned data to train the model, improve the model with grid search and saved the model to a pickle file (_classifer.pkl_).

_classifier.pkl_ already exists but the above command will still run and replace the file will same information.

_**Screenshot 4**_

![train_classifier_1](img/train_classifier_1.jpg)

It took me around **4 minutes** to train the classifier with grid search.

When the models is saved, it will look something like this.

<a id='acc'></a>

**_Screenshot 5_**

![train_classifier_2.jpg](img/train_classifier_2.jpg)

<a id='starting'></a>

### 3.3. Starting the web app

Now that we have cleaned the data and trained our model. Now it's time to see the prediction in a user friendly way.

**Go the app directory** and run the following command:

<a id='com'></a>

```bat
python run.py
```

This will start the web app and will direct you to a URL where you can enter messages and get classification results for it.

**_Screenshot 6_**

![web_app](img/web_app.jpg)

<a id='conclusion'></a>

## 4. Conclusion

Some information about training data set as seen on the main page of the web app.

**_Screenshot 7_**

![genre](img/genre.jpg)

**_Screenshot 8_**

![dist](img/dist.jpg)

As we can see the data is highly imbalanced. Though the accuracy metric is [high](#acc) (you will see the exact value after the model is trained by grid search, it is ~0.94), it has a poor value for recall (~0.6). So, take appropriate measures when using this model for decision-making process at a larger scale or in a production environment.

<a id='files'></a>

## 5. Files

<pre>
.
├── app
│   ├── run.py------------------------# FLASK FILE THAT RUNS APP
│   └── templates
│       ├── go.html-------------------# CLASSIFICATION RESULT PAGE OF WEB APP
│       └── master.html---------------# MAIN PAGE OF WEB APP
├── data
│   ├── DisasterResponse.db-----------# DATABASE TO SAVE CLEANED DATA TO
│   ├── disaster_categories.csv-------# DATA TO PROCESS
│   ├── disaster_messages.csv---------# DATA TO PROCESS
│   └── process_data.py---------------# PERFORMS ETL PROCESS
├── media-----------------------------# IMAGES FOR USE IN README
├── models
│   ├── train_classifier.py-----------# PERFORMS CLASSIFICATION TASK
    └── classifier.pkl----------------# A SAVED MODEL
</pre>

<a id='req'></a>

## 6. Software Requirements

This project uses **Python 3.6.6** and the necessary libraries are mentioned in _requirements.txt_.
The standard libraries which are not mentioned in _requirements.txt_ are _collections_, _json_, _operator_, _pickle_, _pprint_, _re_, _sys_, _time_ and _warnings_.

<a id='credits'></a>

## 7. Credits

Thanks <a href="https://www.udacity.com" target="_blank">Udacity</a> for letting me use their logo as favicon for this web app.

Another <a href="https://medium.com/udacity/three-awesome-projects-from-udacitys-data-scientist-program-609ff0949bed" target="_blank">blog post</a> was a great motivation to improve my documentation. This post discusses some of the cool projects from <a href="https://in.udacity.com/course/data-scientist-nanodegree--nd025" target="_blank">Data Scientist Nanodegree</a> students. This really shows how far we can go if we apply the concepts learned beyond the classroom content to build something that inspire others.