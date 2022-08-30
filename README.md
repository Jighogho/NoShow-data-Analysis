
# Project: Investigate a Dataset - [No Show Appointment]

## Table of Contents
<ul>
<li><a href="#intro">Introduction</a></li>
<li><a href="#wrangling">Data Wrangling</a></li>
<li><a href="#eda">Exploratory Data Analysis</a></li>
<li><a href="#conclusions">Conclusions</a></li>
</ul>

<a id='intro'></a>
## Introduction

### Dataset Description 

This no show appointment dataset (extracted from [Kaggle](https://www.kaggle.com/joniarroba/noshowappointments)) collects information from 110,526 medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. A number of characteristics about the patient are included in each row.
1. ‘Scheduled Day’ tells us on what day the patient set up their appointment.
2. ‘Neighborhood’ indicates the location of the hospital.
3. ‘Scholarship’ indicates whether or not the patient is enrolled in Brasilian

Others characteristics includes the patient's:
4. Gender
5. AppointmentDay
6. Age

and if the patient has:
7. Hypertension
8. Diabetes
9. Alcoholism
10. Handcap

and if the patient recieved SMS or not
11. SMS_received

and also whether the patient showed up for the appointmet or not
12. No-show (Dependent variable)


### Question(s) for Analysis
We are trying to answer the questions:
* What factors can help us predict if a patient will show up for their scheduled appointment?
* Is the system or personal attributes to blame for no shows ?

The report distinguishes between two groups of factors that might have significance: those related to the system, and personal attributes of the patients themselves.


```python
# Set up import statements for all of the packages that I used
import pandas as pd
import numpy as np
import seaborn as sns
import collections
from datetime import datetime
import matplotlib.pyplot as plt
% matplotlib inline
```


```python
# Upgrade pandas to use dataframe.explode() function. 
!pip install pandas
!pip install numpy
!pip install --upgrade seaborn
!pip install datetime
!pip install matplotlib
```

<a id='wrangling'></a>
## Data Wrangling

In this section of the report, let's load in the data, check for cleanliness, and then trim and clean the dataset for analysis. 

First, changing directory to where my dataset lies


```python
cd Database_No_show_appointments
```

Next, Importing my dataset and using the .info() method to het isight on how the dataset looks. We see the column header names, number of entries, and their corresponding data types. We also get to peep into how the etries look like. 


```python
df_nsa=pd.read_csv('noshowappointments-kagglev2-may-2016.csv', sep=',')
df_nsa.info()
```

### General Properties


```python
df_nsa.info
```

Viewing the data in the dataframe under their corresponding header, we have that there are 110526 rows and 14 columns entries


```python
df_nsa.head
```

This method below helps us view the table with their respective header and a few sample data


```python
#This shows the number of unique values in each column

df_nsa.nunique()
```


```python
# This helps us view the index number and label for each column and understand the values i each column

for i, v in enumerate(df_nsa.columns):
    print(i, v)
```


```python
# We will now like to inspect the data set for nulls and duplicates

df_nsa.isnull().sum()
```


```python
df_nsa.duplicated().sum()
```

There are no empty cells ad duplicates. This is really great as we ca work with the data as it is. Let's look further into the data, check unique values for each column.


```python
df_nsa.nunique()
```

We first notice that there are many Neighbourhood values in our data, this makes it less useful in our analysis.

Also, most columns are binary variables (basically variables that take only two values), including our dependent variable, No-Show. This will have an effect on our Exploratory Data Analysis. What this meas is that Pie and bar charts will be more useful than histograms in visualizing and understanding the dataset.

According to Kaggle Handicap should have 2 values (0 or 1) but we noticed that Handcap have 5 unique values. Let's investigate that further. We can define a function for more insights


```python
# function to get unique values
def unique(data):
 
    unique = []
     
    for v in data:
        if v not in unique:
            unique.append(v)

    for v in unique:
        print(v)
```


```python
unique(df_nsa['Handcap'])
```

From this, we see that Handcap takes in 5 range of values 0-4. We can deduce one of two things from this:
1. There are 4 grades of handicaps 1,2,3,4 and 0 indicates none in the system or
2. There was input error

We can get more insights as to the categories of these inputs 0-4


```python
collections.Counter(df_nsa['Handcap'])
```

So we see that the categories of inputs with 2, 3, and 4 are less than 200 (183 + 13 + 3= 199), so the probability that this was an input error is tending towards 1. We can now statistically describe our dataset;


```python
df_nsa.describe()
```

We can see something odd here. We do not expect to have a egatve age but we see that the minimum age is less than 0, so let's investigate this:


```python
collections.Counter(df_nsa['Age']<0)
```

Okay, so that is one age value is less than zero, obviously an error.


## Data Cleaning

Now that we have wrangled our data, we know what errors to clean. So let's start by removing the abnormality with the "Age" column and fixing the errors with the "Handcap" column. "Age" has only one input error of -1, we will replace it with 0:


```python
df_nsa.Age.replace(to_replace=-1, value=0, inplace=True, limit=None, regex=False, method='pad')
collections.Counter(df_nsa['Age']<0)
```

Great, we just successfully fixed the input abnormality with "Age" so that there are no more cells uder the "Age" column with values less than 0. Now to fix the "Handcap" column, since 1 stands for handicaps = True, these positive errors more likely should have been 1. Although changing these inputs to 1 will increase the number of people in this category by 10%; this will still represent very a small percentage of the whole dataset (2%), thereby having only a little effect.


```python
df_nsa.Handcap.replace(to_replace=[2,3,4], value=1, inplace=True, limit=None, regex=False, method='pad')
collections.Counter(df_nsa['Handcap']>1)
```


```python
collections.Counter(df_nsa['Handcap'])
```

We just successfully replaced values greater than 1 with 1. Hence, values in the "Handicap" column are either 0 or 1. Before continuing our cleaning, we need to think about what we need from the dataset. The report explores 'No-show' as a dependent variable. As for the independent variables, they can be classified into two groups:

1. Variables related to the system: (Scholarship, SMS_received, if ScheduledDay == AppointmentDay).
2. Variables related to the patient: (Age, Gender, Medical conditions like Hypertension, Diabetes, Alcoholism, Handcap).

This requires creating a new variable to check if ScheduledDay == AppointmentDay. We can use datetime library, but since both 'ScheduledDay' and 'AppointmentDay' columns are in ISO 8601 standard format, it's much simpler to use string slicing.


```python
# first, creating lists of ScheduledDay and AppointmentDay series
ScheduledDay = list(df_nsa['ScheduledDay'])
AppointmentDay = list(df_nsa['AppointmentDay'])

# next, combinig both lists
dates = list(zip(ScheduledDay, AppointmentDay))
dates[:10]
```

We ca ow create our desired list of bool ad the covert to 0 ad 1


```python
same_testday = []
for sday, aday in dates:
    sameday = sday[5:10] == aday[5:10]
    same_testday.append(sameday)
same_testday[:10]
```


```python
samedate = []
for v in same_testday:
    if v == True:
        v = 1
    else:
        v = 0
    samedate.append(v)
samedate[:10]
```

Perfect, we now know which appointment day ad Sceduled day coincide. So we can now append samedate to df_nsa, we need to create a numpy array, then append it to our dataframe df_nsa


```python
samedate = np.array(samedate)
df_nsa['SameDate'] = samedate
df_nsa['SameDate']
```

Great, we need to do just a few more things. Drop some colums we don't need and lower the string of all others.



```python
df_nsa.info
```


```python
df_nsa= df_nsa.rename(columns=str.lower)
```

Rename 'no-show' to "noshow" and 'sms_received' to "smsreceived"


```python
df_nsa.rename(columns={'no-show':'noshow'}, inplace=True)
```


```python
df_nsa.rename(columns={'sms_received':'smsreceived'}, inplace=True)
```


```python
df_nsa.drop(['patientid', 'appointmentid', 'scheduledday', 'appointmentday','neighbourhood'], axis=1, inplace=True)
```


```python
cols = df_nsa.columns.tolist()
cols

```

Lets change values in 'noshow' from No and Yes, and in 'gender' from F and M to 0 and 1 respectively, to enable us explore the data properly


```python
df_nsa.noshow.replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True, limit=None, regex=False, method='pad')
df_nsa.gender.replace(to_replace=['F', 'M'], value=[0, 1], inplace=True, limit=None, regex=False, method='pad')
```


```python
#Rearragig the colums to start with our depedet variable 'noshow'
cols = ['noshow', 'gender', 'age', 'hypertension', 'diabetes', 'alcoholism', 'handcap',
 'scholarship', 'smsreceived','samedate']
df_nsa = df_nsa[cols]
df_nsa.head()
```

<a id='eda'></a>
## Exploratory Data Analysis

Now that we've trimmed and cleaned our data, we're ready to move on to exploration. Our statistical Computation and visualizations addresses our research questions posed in the Introduction section. It's important to note that this analysis focus is on the correlation between our variables. This is not enough to assume there is a causal relation between. Further studies using inferential statistics is required for that.

**The major questions that this investigation answers are:**

* *What factors can help us predict if a patient will show up for their scheduled appointment?*
* *Is the system or personal attributes to blame for no shows?*


The main problem that this report addresses is the high rate of no_show patient in the Brazilian healthcare system. So let's first explore our dependent variable noshow.

First, let's insight to noshow values. 



```python
absent = df_nsa.noshow == 1
present = df_nsa.noshow == 0
```


```python
# plot the pie chart
plt.pie(df_nsa['noshow'].value_counts(), labels = ['No', 'Yes'], colors=['tab:purple', 'tab:red'], 
        startangle=90, shadow = True, explode = (0, 0.1),
        radius = 1.0, autopct = '%1.1f%%')
# Title
plt.title('Missed the Appointment?', fontweight="bold")

# Display the plot
plt.show()
```

We see that more than 20% missed (were absent from) their appointment. Our aim is to see if we can find out what influenced this. Let us first check the correlation matrix:


```python
cMatrix = df_nsa.corr()
sns.set(rc={'figure.figsize':(15,8)})
sns.heatmap(cMatrix, annot=True)
plt.show()
```

We can now try to answer our research questions

### Research Question 1 (What factors can help us predict if a patient will show up for their scheduled appointment?)

We will investigate each variable: gender, age, hypertension, diabetes, alcoholism, hadcap, scholarship, sms_received and samedate.

### 1. Gender
Patients are either Male (1) or Female (0)

Male patients totals to 35% and female patients accouts for 65% of the dataset as shown in the pie chart below:


```python
plt.pie(df_nsa['gender'].value_counts(), labels = ['Female', 'Male'], colors=['firebrick', 'purple'], 
        startangle=270, shadow = True, explode = (0, 0.1),
        radius = 1.0, autopct = '%1.0f%%')

plt.title('Gender', fontweight="bold")

plt.show()
```


```python
df_nsa['noshow'].corr(df_nsa['gender'])
```

We see that there is no correlation between noshow and gender. The histogram below shows that gender is not a factor to whether or not a patient will have a noshow.


```python
ax = sns.countplot(x="gender", hue="noshow", data=df_nsa)
ax.set(title='Gender and Absent from appointment')
plt.xticks([0, 1], ['Female', 'Male'])
ax.set_xlabel('Gender')
L=plt.legend()
L.get_texts()[0].set_text('Present')
L.get_texts()[1].set_text('Absent')
plt.show()
```

Let's move on to the next factor, 'age'.

### 2. Age
Patients ages lie between 0 and 115 years old. We see this through our .describe() method. 'age' is our only numerical variable in the dataset. Let's start by visualizing its histogram.


```python
ax= df_nsa['age'].hist();
ax.set(title='Age Histogram')
```

The number of patients between the age of 0 ad 10 are cosiderable larger than others. We can investigate this further;



```python
collections.Counter(df_nsa['age']<=10)
```

Let's have a look at a Age box Plot for this data


```python
fig, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(df_nsa['age'])
```


```python
df_nsa['noshow'].corr(df_nsa['age'])

```

There seems to be no correlation between the age parameter as a whole and those who missed their appointement. Hold on let's take a closer look at the age data and see if there are correlations betwee groups of ages and noshow.


```python
ax = sns.countplot(x="age", hue="noshow", data = df_nsa)
ax.set(title='Age and Absent from appointment')
plt.xticks([10, 20, 40, 60, 80, 100, 115], ['Children', 'Teens', 'Youths', 'Adults', 'Elders', 'Old-Elders', 'Old'])
ax.set_xlabel('Age')
L=plt.legend()
L.get_texts()[0].set_text('Present')
L.get_texts()[1].set_text('Absent')
plt.show()
```


```python
# boxplot for each group
ax=sns.boxplot( x ="noshow", y="age", data = df_nsa)
ax.set(xlabel="no show", ylabel="Age")
plt.title('Box Plot Age Comparison');
plt.show()

```


```python
[df_nsa.age[absent].mean(), df_nsa.age[present].mean()]
```

From the above boxplot and code, we find that patients who missed their appointments are younger (on average) than those who showed.

### 3. Hypertension 
19.7% of patients had hypertension, but there is still no correlation with noshow.


```python
plt.pie(df_nsa['hypertension'].value_counts(), labels = ['No', 'Yes'], colors=['green', 'red'], 
        startangle=90, shadow = True, explode = (0, 0.1),
        radius = 1.0, autopct = '%1.1f%%')
  
plt.title('Has Hypertension?', fontweight="bold")
     
plt.show()
```


```python
df_nsa['noshow'].corr(df_nsa['hypertension'])
```


```python
ax = sns.countplot(x="hypertension", hue="noshow", data=df_nsa)
ax.set(title='Has hypertension and Absent from appointment')
plt.xticks([0, 1], ['No Hypertension', 'Has Hypertension'])
ax.set_xlabel('Hypertension')
L=plt.legend()
L.get_texts()[0].set_text('Present')
L.get_texts()[1].set_text('Absent')
plt.show()
```

### 4. Diabetes
7.2% of patients had diabetes, but there is still no correlation with noshow.


```python
plt.pie(df_nsa['diabetes'].value_counts(), labels = ['No', 'Yes'], colors=['green', 'red'], 
        startangle=90, shadow = True, explode = (0, 0.1),
        radius = 1.0, autopct = '%1.1f%%')
  
plt.title('Has Diabetes?', fontweight="bold")
     
plt.show()
```


```python
df_nsa['noshow'].corr(df_nsa['diabetes'])
```


```python
ax = sns.countplot(x="diabetes", hue="noshow", data=df_nsa)
ax.set(title='Has Diabetes and Absent from appointment')
plt.xticks([0, 1], ['No Diabetes', 'Has Diabetes'])
ax.set_xlabel('Diabetes')
L=plt.legend()
L.get_texts()[0].set_text('Present')
L.get_texts()[1].set_text('Absent')
plt.show()

```

### 5. Alcoholism

3% of patients are alcholic, but there is still no correlation with noshow. 


```python
plt.pie(df_nsa['alcoholism'].value_counts(), labels = ['No', 'Yes'], colors=['green', 'red'], 
        startangle=90, shadow = True, explode = (0, 0.1),
        radius = 1.0, autopct = '%1.1f%%')
  
plt.title('Is Alcholic?', fontweight="bold")
     
plt.show()
```


```python
df_nsa['noshow'].corr(df_nsa['alcoholism'])
```


```python
ax = sns.countplot(x='alcoholism', hue="noshow", data=df_nsa)
ax.set(title='Has Diabetes and Absent from appointment')
plt.xticks([0, 1], ['Not Alcholic', 'Alcoholic'])
ax.set_xlabel('Alcholism')
L=plt.legend()
L.get_texts()[0].set_text('Present')
L.get_texts()[1].set_text('Absent')
plt.show()
```


```python
ax= df_nsa['alcoholism'].hist();
ax.set(title='Alcholic Histogram')
```

### 6. Handcap

Handicap accounts for 2% of patients, but there is still no correlation with noshow.


```python
plt.pie(df_nsa['handcap'].value_counts(), labels = ['No', 'Yes'], colors=['green', 'red'], 
        startangle=90, shadow = True, explode = (0, 0.1),
        radius = 1.0, autopct = '%1.1f%%')
  
plt.title('Is Handicap?', fontweight="bold")
     
plt.show()
```

### 7. Scholarship

It provides financial aid to poor families.

scholarship = 1 for patients who receive funds from this program. This totals to 9.8% of the dataset as shown in the pie chart below:


```python
plt.pie(df_nsa['scholarship'].value_counts(), labels = ['No', 'Yes'], colors=['green', 'red'], 
        startangle=270, shadow = True, explode = (0, 0.1),
        radius = 1.0, autopct = '%1.1f%%')
  
plt.title('Is Handicap?', fontweight="bold")
     
plt.show()
```

### 8. Sms Recieved 

There is a weak correlative relationship between sms recieved and showing up to the appointment.


```python
plt.pie(df_nsa['smsreceived'].value_counts(), labels = ['No', 'Yes'], colors=['tab:blue', 'tab:pink'], 
        startangle=90, shadow = True, explode = (0, 0.1),
        radius = 1.0, autopct = '%1.1f%%')

plt.title('SMS Received?', fontweight="bold")
    
plt.show()
```


```python
df_nsa['noshow'].corr(df_nsa['smsreceived'])
```

We see here that people that didn't recieve SMS are more likely to show up at their appointment than those that received sms.


```python
ax = sns.countplot(x="smsreceived", hue="noshow", data=df_nsa)
ax.set(title='SMS Notifications and Absent from Appointments')
plt.xticks([0, 1], ['No SMS', 'Received SMS'])
ax.set_xlabel('SMS Received')
L=plt.legend()
L.get_texts()[0].set_text('Present')
L.get_texts()[1].set_text('Absent')
plt.show()
```

### 9.Same date

Are people that schedule their appointment for the same day likely to miss it. Let's find out: 

34.9% got their appointment on the same day.


```python
plt.pie(df_nsa['samedate'].value_counts(), labels = ['No', 'Yes'], colors=['tab:pink', 'tab:purple'], 
        startangle=270, shadow = True, explode = (0, 0.1),
        radius = 1.0, autopct = '%1.1f%%')
  
plt.title('Scheduled for the Same Day?', fontweight="bold")
     
plt.show()
```


```python
df_nsa['noshow'].corr(df_nsa['samedate'])
```

There is a negative correlation between `same date` and `noshow`. Though the correlation is not so strong, it is the strongest correlation we have gotten so far. This correlation tells us that patients who schedule their appointments for the same date are less likely to be absent.


```python
ax = sns.countplot(x="samedate", hue="noshow", data=df_nsa)
ax.set(title='Missed Appointments depending on when Scheduled for')
plt.xticks([0, 1], ['Different date', 'Same date'])
ax.set_xlabel('Schedule Day vs Appointment Day')
L=plt.legend()
L.get_texts()[0].set_text('Present')
L.get_texts()[1].set_text('Absent')
plt.show()
```



### Research Question 2  (Is the system or personal attributes to blame for no shows?)
Let's split this session into two.

`a. Is the system to blame for no shows?`

`b. Are personal attributes to blame for no shows?`

Recall we grouped the variables into Personal attributes and system properties. Just to recap:

i. *Variables related to the system:* (Scholarship, SMS received, samedate [where ScheduledDay == AppointmentDay]).

ii. *Variables related to the patient:* (Age, Gender, Medical conditions like Hypertension, Diabetes, Alcoholism, Handcap).

### Is the system to blame for no shows? 
Let's review and explore more insights on the Variables related to the system. We have looked at the Scholarship, SMS received, and same date scheduled appointment properties individually. The prominent features are SMS received and same date. Let's look at what a combination of both tells us about the data.

**Same date and SMS received:**

We see that we patients that received sms are those that didn't schedule their appointment for the same day. The system did not see a need to send a notification and their was still a huge turn up. We an see this because of the empty dataframe for the truth values of both variables So there is no correlation between these two variables and showing up, `noshow` 


```python
sick= df_nsa.groupby(['samedate', 'smsreceived'], as_index=False)['noshow'].mean()
sick.head()
```


```python
sick= df_nsa.groupby(['samedate', 'smsreceived'], as_index=False)['noshow'].corr()
sick.head()
```

### Are personal attributes to blame for no shows?
Let's review and explore more insights on the variables related to the personal attributes. We have looked at the Age, Gender, Medical conditions like Hypertension, Diabetes, Alcoholism, Handcap individually. The prominent features are younger Age. The correlation of hypertension and diabetes with noshow individually were insignificant. Let's look at what a combination of both tells us about the data.

**Hypertension and diabetes:**
Actually, if a patient suffer from both illnesses, he much less likely to miss the appointment.


```python
sick= df_nsa.groupby(['diabetes', 'hypertension'], as_index=False)['noshow'].mean()
sick.head()
```


```python
colors = ['red', 'white', 'blue']
sickness_means = df_nsa.groupby(['diabetes', 'hypertension'], as_index=False)['noshow'].mean()
sickness_means.plot(kind='bar', title='Average diabetic patients by hypertension', color=colors, alpha=.7);
plt.xlabel('Diabetes', fontsize=18)
plt.ylabel('Hypertension', fontsize=18)
```




<a id='conclusions'></a>
## Conclusions

### Findings and Results

***Regarding factors can help us predict if a patient will show up for their scheduled appointment?***

This report examined a dataset of Brazilian patients and thier likelihood to show up for a scheduled appointment. We tried to find out what factors correlates with being absent from appointments, what is referred to as `noshow`.

The most important factor was whether the patient scheduled his appointment for the same day of booking. This factor has a negative correlation with `no_show` cases. This means that patients where more likely to be present for same date appointment, that is less no_shows when the patient was scheduled on the same day.

Also, the younger the patient the more likely there will be a no show to the appointment.

***Regarding the system*** 

Here, we found that sending SMS before hand didn't help. It was actually weakly positively correlated with absent cases (no show). This means that those who received an SMS were slightly more likely to be absent on the appointment day. Scholarship program benefits didn't seem to have any useful correlation with no_show cases.

***Regarding Personal attributes***

This did not correlate with no_shows (absence). But when we partition the ages, we noticed that patients who missed their appointments were younger in general of those who didn't. 


It is also noted that having multiple illnesses seems to correlate with less no_shows.


### Limitation


The dependent variable noshow is a binary variable, infact most variables in the dataset are binary (take in only two values). This limits the statistical methods that can be used to analyze the data.

It's important to note that this analysis focus is on the correlation between the variables. We cannot assume there is a causal relation between them. There is need for further analysis using inferential statistics. Especially since most correlations were weak because of the nature of the data variables.

One would think that receiving an SMS notification prior to the appointment will improve the situation but the results were counter-intuitive. Wwe in fat found that there is a slight higher chance of missing the appointment for those who received an SMS. 

The nature of the data might prevent further investigation.


```python
from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])
```
# NoShow-data-Analysis
