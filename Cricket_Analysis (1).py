#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


df=pd.read_excel(r"C:/Users/harsh/Desktop/cricket.xlsx")
print(df)


# In[3]:


df.head(10)


# In[4]:


df=df.rename(columns={"Mat":"Matches","BF":"Balls_Faced","SR":"Strike_rate"})
df.head()


# In[5]:


df.isnull().any()


# In[6]:


df.isnull().sum()


# In[7]:


df[df["Balls_Faced"].isna()==1]


# In[8]:


df["Balls_Faced"]=df["Balls_Faced"].fillna(0)
df["Strike_rate"]=df["Strike_rate"].fillna(0)
df.head(20)


# In[9]:


df[df["Player"].duplicated()==1]


# In[10]:


df=df.drop_duplicates()


# In[11]:


df.head(20)


# In[12]:


df["Span"].str.split(pat = "-")


# In[13]:


df["Start_year"]=df["Span"].str.split(pat = "-").str[0]


# In[14]:


df["Final_year"]=df["Span"].str.split(pat = "-").str[1]


# In[15]:


df


# In[16]:


df.drop(["Span"],axis=1,inplace=True)


# In[17]:


df.head()


# In[18]:


df["Player"].str.split(pat="(")


# In[19]:


df["Country"]=df["Player"].str.split(pat="(").str[1]
df["Country"]=df["Country"].str.split(pat=")").str[0]


# In[20]:


df["Player"]=df["Player"].str.split(pat="(").str[0]


# In[21]:


df.head()


# In[22]:


df.info()


# In[23]:


df["Balls_Faced"].str.split(pat = "+")


# In[24]:


df["Balls_Faced"]=df["Balls_Faced"].str.split(pat = "+").str[0]


# In[25]:


df


# In[26]:


df["HS"]=df["HS"].str.split(pat = "*").str[0]


# In[27]:


df


# In[28]:


df.info()


# In[29]:


df.isnull().any()


# In[30]:


df[df["Balls_Faced"].isna()==1]


# In[31]:


df["Balls_Faced"]=df["Balls_Faced"].fillna(0)


# In[32]:


df[df["Balls_Faced"].isna()==1]


# In[33]:


df= df.astype({"HS":"int","Balls_Faced":"int","Start_year":"int","Final_year":"int"})
df.info()


# In[34]:


df["carrer_length"]=df["Final_year"]-df["Start_year"]


# In[35]:


plt.subplot(2, 2, 4)
sns.histplot(df["Strike_rate"], bins=50, kde=True, color='purple')
plt.title('Distribution of Strike Rate')

plt.tight_layout()
plt.show()


# In[36]:


df.head()


# In[37]:


get_ipython().run_line_magic('matplotlib', 'inline')
df.hist(bins=50, figsize=(20,15))
plt.show()


# In[38]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[40]:


# Check the lengths of x and y
x= df[['Matches','Strike_rate']]
y = df['Runs']
print(f"Length of x: {len(x)}")
print(f"Length of y: {len(y)}")

# Check for missing values
print(df[['Matches', 'Strike_rate', 'Runs']].isnull().sum())


# In[50]:


df_clean = df.dropna(subset=['Matches', 'Strike_rate', 'Runs'])

# Set up the predictor and response variables
X = df_clean[['Matches', 'Strike_rate']]
Y = df_clean['Runs']

# Ensure that X and Y now have the same length
print(f"Length of X: {len(X)}")
print(f"Length of Y: {len(Y)}")


# In[52]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=23)


# In[45]:


X_train = np.array(X_train).reshape(-1,1)
X_testn=np.array(X_test).reshape(-1,1)


# In[53]:


lr = LinearRegression()
lr.fit(X_train, Y_train)


# In[55]:


predictions = lr.predict(X_train)
print(predictions)


# In[59]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train['Matches'], y=Y_train, color="blue", label="Actual")
sns.lineplot(x=X_train['Matches'], y=predictions, color="red", label="Predicted")
plt.xlabel('Matches')
plt.ylabel('Runs')
plt.title('Linear Regression - Runs vs Matches with Strike Rate')
plt.legend()
plt.show()


# In[63]:


def predict_runs(player_name, matches_played):
    player_data = df[df["Player"].str.contains(player_name, case=False, na=False)]
    
    if not player_data.empty:
        # Predict runs based on the matches played
        predicted_runs = lr.predict([[matches_played]])
        print(f"Predicted runs for {player_name} who played {matches_played} matches: {predicted_runs[0]:.2f}")
    else:
        print(f"Player {player_name} not found in the dataset.")

# Example usage
player_name = input("Enter the player's name: ")
matches_played = int(input("Enter the number of matches played: "))

predict_runs("SR Tendulkar", 200)


# In[64]:


df['Player'].isnull()


# In[65]:


df[df['Player'].str.contains('SR Tendulkar' , case = False , na = False)]

