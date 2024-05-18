import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from io import StringIO

# Title and description
st.title('Screen Time and Health Analysis Dashboard')
st.write("""
This dashboard provides an analysis of survey data on screen time, sleep patterns, and their impact on health and social interactions.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r'Screen Time Analysis Survey.csv')
    df.drop(['Timestamp'], axis=1, inplace=True)
    df.rename(columns={
        'What is your age?':'Age', 
        'What is your gender?':'Gender',
        'What is your education level?':'Education',
        'On an average weekday, how many hours do you spend looking at screens (computers, smartphones, tablets, TVs, etc.) for non-academic purposes (e.g., entertainment, social media, gaming)?':'ST_Weekday', 
        'On an average weekend day, how many hours do you spend looking at screens for non-academic purposes?':'ST_Weekend', 
        'Which of the following activities do you engage in on screens during your leisure time?': 'Leisure_ST_Act', 
        'How often do you engage in this activity?':'Average_Distraction', 
        'Do you experience any of the following health issues due to a prolonged screen time?':'Health_Effects',
        'Are you aware of the negative effects of excessive screen time on health (e.g., eye strain, sleep disturbances, sedentary lifestyle, mental health)?':'Health_awarness', 
        'What devices do you primarily use for leisure screen time?':'Devices_Used_In_Leisure',  
        'On average, how many hours of sleep do you get on a typical weekday night?':'Ave_Weekday_Sleep',
        'On average, how many hours of sleep do you get on a typical weekend night?':'Ave_Weekend_Sleep',
        'Do you use any corrective eyewear? (specs, contact lenses, etc)':'Eyewear_User',
        'Has your screen time affected your face-to-face social interactions with friends and family?':'Interaction_Affects',
        'Do you actively try to limit your screen time to mitigate these negative effects?':'Mitigate_Screen_time'
    }, inplace=True)
    df.replace(np.nan, '0', inplace=True)
    return df

df = load_data()

# Show data
if st.checkbox('Show raw data'):
    st.write(df.head(10))

# Data information
st.subheader('Data Information')
st.write(f"Number of rows: {df.shape[0]}")
st.write(f"Number of columns: {df.shape[1]}")
st.write("Columns:")
st.write(df.columns.tolist())

# Basic info and statistics
st.subheader('Basic Information')
buffer = StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.subheader('Statistical Summary')
st.write(df.describe())

# Check for null values
st.subheader('Null Values')
st.write(df.isnull().sum())

# Correlation analysis
st.subheader('Correlation Analysis')
numeric_df = df.select_dtypes(include=[np.number])
st.write(numeric_df.corr())

fig, ax = plt.subplots()
sns.heatmap(numeric_df.corr(), ax=ax, annot=True, cmap='coolwarm')
st.pyplot(fig)

# Boxplot for checking outliers
st.subheader('Outlier Analysis - Weekday Sleep')

# Print the columns to ensure 'Ave_Weekday_Sleep' exists
st.write("Columns in DataFrame:", df.columns.tolist())

if 'Ave_Weekday_Sleep' in df.columns:
    fig, ax = plt.subplots()
    sns.boxplot(y=df['Ave_Weekday_Sleep'], ax=ax)
    st.pyplot(fig)

    avg_sleep = df['Ave_Weekday_Sleep'].mean()
    min_sleep = df['Ave_Weekday_Sleep'].min()
    max_sleep = df['Ave_Weekday_Sleep'].max()
    std_dev = df['Ave_Weekday_Sleep'].std()
    upper_limit = avg_sleep + 3*std_dev
    lower_limit = avg_sleep - 3*std_dev

    st.write("The Average weekday sleep of the respondents was found to be: ", avg_sleep, "hrs")
    st.write("The Minimum weekday sleep of the respondents was found to be: ", min_sleep, "hrs")
    st.write("The Maximum weekday sleep of the respondents was found to be: ", max_sleep, "hrs")
    st.write("Upper limit for outliers: ", upper_limit)
    st.write("Lower limit for outliers: ", lower_limit)

    # Replace outliers
    df.loc[df['Ave_Weekday_Sleep'] >= upper_limit, 'Ave_Weekday_Sleep'] = upper_limit
    fig, ax = plt.subplots()
    sns.boxplot(y=df['Ave_Weekday_Sleep'], ax=ax)
    st.pyplot(fig)
else:
    st.write("Column 'Ave_Weekday_Sleep' does not exist in the DataFrame.")

# Group by age group and calculate mean weekday sleep
st.subheader('Average Weekday Sleep by Age Group')
if 'Age' in df.columns and 'Ave_Weekday_Sleep' in df.columns:
    age_group_means = df.groupby('Age')['Ave_Weekday_Sleep'].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    age_group_means.plot(kind='bar', color='skyblue', edgecolor='k', ax=ax)
    ax.set_title('Average Weekday Sleep by Age Group')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Average Weekday Sleep (hours)')
    ax.grid(True)
    st.pyplot(fig)
else:
    st.write("Columns 'Age' and/or 'Ave_Weekday_Sleep' do not exist in the DataFrame.")

# Respondents with less than 6 hours of weekday sleep
st.subheader('Respondents with Less than 6 Hours of Weekday Sleep')
if 'Ave_Weekday_Sleep' in df.columns:
    filtered_df = df[df['Ave_Weekday_Sleep'] < 6]
    st.write(filtered_df.head())

    st.write('Health issues, eyewear use, and social interaction of such respondents')
    if {'Health_Effects', 'Eyewear_User', 'Interaction_Affects'}.issubset(df.columns):
        st.write(filtered_df[['Health_Effects', 'Eyewear_User', 'Interaction_Affects']].head())
    else:
        st.write("One or more of the columns 'Health_Effects', 'Eyewear_User', 'Interaction_Affects' do not exist in the DataFrame.")

    # Scatter plot of sleep time and health issues
    st.subheader('Impact of Sleep Time on Health Issues')
    if 'Health_Effects' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(filtered_df['Ave_Weekday_Sleep'], filtered_df['Health_Effects'], alpha=0.5)
        ax.set_title('Impact of Sleep Time on Health Issues')
        ax.set_xlabel('Weekday Sleep (hours)')
        ax.set_ylabel('Health Issues')
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.write("Column 'Health_Effects' does not exist in the DataFrame.")
else:
    st.write("Column 'Ave_Weekday_Sleep' does not exist in the DataFrame.")

# Chi-square test for screentime and social interaction (Weekday)
st.subheader('Impact of Screentime on Social Interaction (Weekday)')
if 'ST_Weekday' in df.columns and 'Interaction_Affects' in df.columns:
    contingency_table = pd.crosstab(df['ST_Weekday'], df['Interaction_Affects'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    st.write("p-value is: ", p)
    if p < 0.05:
        st.write("There is a significant association between average weekday screentime and social interaction.")
    else:
        st.write("There is no significant association between average weekday screentime and social interaction.")
else:
    st.write("Columns 'ST_Weekday' and/or 'Interaction_Affects' do not exist in the DataFrame.")

# Chi-square test for screentime and social interaction (Weekend)
st.subheader('Impact of Screentime on Social Interaction (Weekend)')
if 'ST_Weekend' in df.columns and 'Interaction_Affects' in df.columns:
    contingency_table = pd.crosstab(df['ST_Weekend'], df['Interaction_Affects'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    st.write("p-value is: ", p)
    if p < 0.05:
        st.write("There is a significant association between average weekend screentime and social interaction.")
    else:
        st.write("There is no significant association between average weekend screentime and social interaction.")
else:
    st.write("Columns 'ST_Weekend' and/or 'Interaction_Affects' do not exist in the DataFrame.")
