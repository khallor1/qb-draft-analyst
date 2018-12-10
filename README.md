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

Technologies used:
BeautifulSoup, re (regex), requests, pandas, scikit-learn, flask, json

Other stuff:

