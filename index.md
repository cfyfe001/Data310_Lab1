## Welcome to my Lab 1 Work!

### The following information corresponds to each question!

# Question 1
*What would be the most commonly used level of measurement if the variable is the temperature of the air?*

For this question, I selected **'interval'** because it is important to note differences in the measured temperature, but it would be meaningless to create a proportion for temperature.

# Question 2
*Write a Python code to import the data file 'L1data.csv' (introduced in Lecture 1) and code an imputation for replacing the NaN values in the "Age" column with the median of the column. The NaN instances are replaced by:*

The code I used was: 
```markdown
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
data = pd.read_csv('drive/MyDrive/Colab Notebooks/L1Data.csv')

imputing_configuration = SimpleImputer(missing_values = np.nan, strategy = 'median')
imp = imputing_configuration.fit(data[["Age"]])
data[["Age"]] = imp.transform(data[["Age"]]).ravel()
data
```
Doing this allowed me to understand that the NaNs in the 'Age' column were replaced with the median age of **21.0**.

# Question 3
*In Bayesian inference the "likelihood" represents:*

I selected **"How probable is the data (evidence) given that our hypothesis is true"** because the notes specify that 'likelihood represents how probable is the evidence given that our hypothesis is true.'

# Question 4
*The main goal of Monte Carlo simulations is to solve problems by approximating a probability value via carefully designed simulations. True/False*

I selected **True** for this question because this ***is*** the goal of Monte Carlo simulations. The notes state that the main goal of Monte Carlo simulations are to solve problems of data science by approximating probability values via carefully designed simulations.

# Question 5
*Assume that during a pandemic 15% of the population gets infected with a respiratory virus while about 35% of the population has some general respiratory symptoms such as sneezing, stuffy nose etc. Assume that approximately 30% of the people infected with the virus are asymptomatic. What is the probability that someone who has the symptom actually has the disease?*

This question is an example of Bayesian Inference. Using the formula provided in the notes, the math for this question is as follows:

```markdown
P(A) = 0.15
P(B) = 0.35
P(B|A) = 1 - 0.3 = 0.7

(P(B|A) * P(A)) / P(B) = 0.3
```
Based on this math, there is a 30% probability that someone who has the symptom actually has the disease. I did 1 - 0.3 for P(B|A) because the problem states that 30% of infected people are asymptomatic, but we are trying to understand the probability of someone ***who has*** symptoms.

# Question 6
*A Monte Carlo simulation should never include more than 1000 repetitions of the experiment. True/False*

I selected **False** for this question because the more repetitions of an experiment, the better results you get. In class, we did experiments going up to 5000 repetitions! Based on these facts, this is **false**.

# Question 7
*One can decide that the number of iterations in a Monte Carlo simulation was sufficient by visualizing a Probability-Iteration plot and determining where the probability graph approaches a horizontal line. True/False*

I selected **True** for this answer because of the graphs we created in class. The code below creates a Monte Carlo simulation with 2000 repetitions concerning the probability of heads/tails for a coin flip. This will create a Probability-Iteration plot.
```markdown
import random
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
 
def coin_flip():
  return random.randint(0,1)
 
listp = []
 
def monte_carlo(n):
  results = 0;
  for i in range(n):
    flip_result = coin_flip()
    results = results + flip_result
 
    prob_val = results/(i+1)
 
    listp.append(prob_val)
 
  return results/n
 
answer = monte_carlo(2000)
  
plt.axhline(y=0.5, color = 'red')
plt.plot(listp)
plt.xlabel("Iterations",color='dodgerblue')
plt.ylabel("Probability",color='seagreen')
```

# Question 8
*Assume we play a slightly bit different version of the original Monte Hall problem such as having four doors one car and three goats. The rules of the game are the same, the contestant chooses one door (that remains closed) and one of the other doors who had a goat behind it is being opened. The contestant has to make a choice as to stick with the original choice or rather switch for one of the remaining closed doors. Write a Python code to approximate the winning probabilities, for each choice, by the means of Monte Carlo simulations. The probability that the contestant will ultimately win by sticking with the original choice is closer to:*

For this question, I used the code listed below:
```markdown
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

import random
import matplotlib.pyplot as plt

doors = ["goat","goat","goat", "car"]

switch_win_probability = []
stick_win_probability = []

plt.axhline(y=0.75, color='red', linestyle='--')
plt.axhline(y=0.25, color='green', linestyle='--')

def monte_carlo(n):

  switch_wins = 0
  stick_wins = 0

  for i in range(n):
     random.shuffle(doors)

     k = random.randrange(4)

     if doors[k] != 'car':
       switch_wins +=1
    
     else:
       stick_wins +=1
    
     switch_win_probability.append(switch_wins/(i+1))
     stick_win_probability.append(stick_wins/(i+1))
    
  plt.plot(switch_win_probability,label='Switch')
  plt.plot(stick_win_probability,label='Stick')
  plt.tick_params(axis='x', colors='navy')
  plt.tick_params(axis='y', colors='navy')
  plt.xlabel('Iterations',fontsize=14,color='DeepSkyBlue')
  plt.ylabel('Probability of Winning',fontsize=14,color='green')
  plt.legend()
  print('Winning probability if you always switch:', switch_win_probability[-1])
  print('Winning probability if you always stick to your original choice:', stick_win_probability[-1])

monte_carlo(5000)
```
After a few trials and increasing the repetitions, I found that your winning probability, if you switched, was around 75% while the winning probability, if you stayed, was about 25%. Due to this, I chose **25%** as my final answer.

# Question 9
*In Python one of the libraries that we can use for generating repeated experiments in Monte Carlo simulations is:*

Looking through the code above allows you to see that the **random** library is vital for the repetitions to occur. Without this library, the Monte Hall code would not work.

# Question 10
*In Python, for creating a random permutation of an array whose entries are nominal variables we used:*

The code from Question 8 also assists with this question. The **doors** variable are an arrary of nominal values. Later in the code, the **random.shuffle()** method is used to alter the order of **doors**. Based on this, it can be concluded that **random.shuffle()** is the correct answer.
