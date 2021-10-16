
# Adult Census Income Prediction

## Table of contents
  * [Demo](#demo)
  * [Overview](#overview)
  * [Project Goal](#project-goal)
  * [Technical Aspect](#technical-aspects)
  * [Installation](#installation)
  * [Bug / Feature Request](#bug-/-feature-request)
  * [Technologies Used](#technologies-used)
  * [License](#license)

## Demo

• Link for web application : https://incomepred.anvil.app

• Working of web application on a PC

![Alt Text](https://drive.google.com/uc?export=download&id=1JgprtQnTNKGx1EC8W9t9JfKd5HfMAWug)

• Working of application on mobile phone

![Alt Text](https://drive.google.com/uc?export=download&id=1wphArRjCBUA1cA8ux8P3SJoZ6__1U-i8)

## Overview

This is a classification problem where we need to predict whether a person earns more than a sum of 50,000 USD anuually or not. This classification task is accomplished by using a CatBoost Classifier trained on the dataset extracted by Barry Becker from the 1994 Census database. The dataset contains about 33k records and 15 features which after all the implementation of all standard techniques like Data Cleaning, Feature Engineering, Feature Selection, Outlier Treatment, etc was feeded to our Classifier which after training and testing, was deployed in the form of a web application.

## Project Goal

This end-to-end project is made as a part of data science internship for [Ineuron.ai](https://ineuron.ai/).

## Technical Aspects

The whole project has been divided into three parts. These are listed as follows :

• Data Preparation : This consists of storing our data into cassandra database and utilizing it, Data Cleaning, Feature Engineering, Feature Selection, EDA, etc.

• Model Development : In this step, we use the resultant data after the implementation of the previous step to cross validate our Machine Learning model and perform Hyperparameter optimization based on various performance metrics in order to make our model predict as accurate results as possible.

• Model Deployment : This step include creation of a front-end using Anvil, Flask and Heroku to put our trained model into production.

## Installation

The Code is written in Python 3.9. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository:
```bash
pip install -r requirements.txt
```

### Run on your Local Machine

To run the flask server on your local machine, just run the following code in your command prompt in the project directory :
```bash
python server.py
```
This will start the run the [server.py](https://github.com/RauhanAhmed/Ineuron_internship/blob/main/server.py) which will also trigger code for [server_app.py](https://github.com/RauhanAhmed/Ineuron_internship/blob/main/server_app.py) because of the use of asynchronous execution (threading) and will connect our ML model to Anvil application UI and will keep the server running till the web page rendered by flask application gets closed but to keep the server running forever, we used the heroku cloud to run our server continuously.

## Bug / Feature Request
If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here](https://github.com/rauhanahmed/ineuron_internship/new) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/rauhanahmed/ineuron_internship/issues/new). Please include sample queries and their corresponding results.

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

• Python Programming language

• Heroku cloud

• Anvil framework

## Appendix

Link for youtube video regarding description of the project : https://youtu.be/KFZXXu3OSSk

Link for App Documentation : https://github.com/RauhanAhmed/Ineuron_internship/tree/main/App%20Documents

## Author

[Rauhan Ahmed Siddiqui](https://github.com/RauhanAhmed)
## License

[MIT](https://choosealicense.com/licenses/mit/)

Copyright 2021 Rauhan Ahmed Siddiqui

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
