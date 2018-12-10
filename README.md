# qb-draft-analyst
CIS192 Final Project (Kieran Halloran)

Methodology:
Since the purpose of this project is to predict the performace of draft prospects the the
quarterback position, I will look up the statistics of all quarterbacks drafted into the
nfl from 1997-2017.  Using these stats, I train various ML models that either classify
the quarterbacks as a success or bust (using the criterion developed by Stanford
students: cs229.stanford.edu/proj2017/final-reports/5231213.pdf), or use a regressor to
predict career NFL stats.  Then, I scrape the data of current top NFL draft qb
prospects, and using these models, predict their performance.  I created a flak app to 
streamline interaction

Modules used:
BeautifulSoup, re (regex), requests, pandas, scikit-learn, flask, json

How to use:
I already scraped the data (via jupyter notebooks), so to use the models and look at 
predictions, simply type
./app.py in your terminal and open http://localhost:5000/ in your browser.

Custom class:
I created a QBModel class that gets imported into app.py so that it can be used by my 
flask app.  It stores the training data as a class variable and returns the results of 
various ML predictions, depending on which function is called.  I added the __str__ magic
method.

Decorator:
I created a timer that appends the time it takes to run a function to its output string.  
I used this so my predictions would show time, kind of like google search.
