import streamlit as st
import base64
import requests
from streamlit_lottie import st_lottie
import json
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import register_matplotlib_converters
import datetime
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Register converters to avoid warnings
register_matplotlib_converters()

# Initial page configuration
st.set_page_config(
    page_title='📊 Comprehensive Pandas Cheat Sheet By Mejbah Ahammad',
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    ds_sidebar()
    ds_body()

# Function to convert image to base64 bytes (for logo)
def img_to_bytes(img_url):
    try:
        response = requests.get(img_url)
        img_bytes = response.content
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    except:
        return ''

# Function to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Sidebar content with enhanced design
def ds_sidebar():
    logo_url = 'https://ahammadmejbah.com/content/images/2024/10/Mejbah-Ahammad-Profile-8.png'
    logo_encoded = img_to_bytes(logo_url)
    
    # Custom CSS for sidebar
    sidebar_style = """
    <style>
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    /* Sidebar header styling */
    .sidebar .sidebar-content h2 {
        color: #FF4B4B;
    }
    /* Sidebar links styling */
    .sidebar .sidebar-content a {
        color: #333333;
        text-decoration: none;
    }
    .sidebar .sidebar-content a:hover {
        color: #FF4B4B;
    }
    </style>
    """
    st.sidebar.markdown(sidebar_style, unsafe_allow_html=True)
    
    # Display logo
    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <a href="https://ahammadmejbah.com/">
                <img src='data:image/png;base64,{logo_encoded}' class='img-fluid' width=100>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.header('🧰 Comprehensive Pandas Cheat Sheet')
    
    st.sidebar.markdown('''
    <small>All-encompassing summary of essential Pandas concepts, functions, and best practices.</small>
    ''', unsafe_allow_html=True)
    
    st.sidebar.markdown('__🔑 Key Libraries__')
    st.sidebar.code('''
$ pip install pandas numpy matplotlib seaborn sqlalchemy openpyxl scikit-learn statsmodels plotly streamlit-lottie
    ''')
    
    st.sidebar.markdown('__💻 Common Commands__')
    st.sidebar.code('''
$ jupyter notebook
$ python script.py
$ git clone https://github.com/ahammadmejbah/Data-Science-Cheat-Sheet
$ streamlit run app.py
    ''')
    
    st.sidebar.markdown('__💡 Tips & Tricks__')
    st.sidebar.code('''
- Use virtual environments
- Version control with Git
- Document your code
- Continuous learning
- Utilize Jupyter Notebooks for exploration
- Optimize memory usage with appropriate data types
- Leverage vectorized operations for performance
- Utilize Pandas' built-in functions for data manipulation
- Handle missing data effectively
- Use descriptive statistics for data insights
- Explore data with visualization tools
- Automate repetitive tasks with functions
- Validate data integrity before analysis
- Use meaningful variable names
- Modularize code for reusability
- Keep up with the latest Pandas updates
    ''')
    
    st.sidebar.markdown('''<hr>''', unsafe_allow_html=True)
    st.sidebar.markdown('''<small>[Comprehensive Pandas Cheat Sheet v1.0](https://github.com/ahammadmejbah/Data-Science-Cheat-Sheet) | Nov 2024 | [Mejbah Ahammad](https://ahammadmejbah.com/)<div class="card-footer">Mejbah Ahammad © 2024</div></small>''', unsafe_allow_html=True)

# Main body of cheat sheet with two-row navigation
def ds_body():
    # Load Lottie animations
    lottie_header = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_ydo1amjm.json")
    lottie_section = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_1pxqjqps.json")
    
    # Header with animation
    st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #FF4B4B;">📊 Comprehensive Pandas Cheat Sheet By Mejbah Ahammad</h1>
        </div>
    """, unsafe_allow_html=True)
    
    if lottie_header:
        st_lottie(lottie_header, height=200, key="header_animation")
    
    # Define Pandas topics and their extended code snippets
    sections_row1 = {
        "📦 Importing & Setup": {
            "Importing Libraries": '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setting display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

# Suppress warnings for clean output
import warnings
warnings.filterwarnings('ignore')
            ''',
            "Reading Data": '''
# Read CSV with specific encoding
df_csv = pd.read_csv('data.csv', encoding='utf-8')

# Read Excel specifying sheet and engine
df_excel = pd.read_excel('data.xlsx', sheet_name='Sheet1', engine='openpyxl')

# Read JSON with normalization for nested structures
df_json = pd.read_json('data.json')
df_normalized = pd.json_normalize(json_data, 'records')

# Read from SQL database
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:password@localhost:5432/mydatabase')
df_sql = pd.read_sql('SELECT * FROM table_name WHERE age > 30', engine)

# Read from HDF5 file
df_hdf = pd.read_hdf('data.h5', key='df_key')

# Read from Parquet file
df_parquet = pd.read_parquet('data.parquet')

# Read from Pickle file
df_pickle = pd.read_pickle('data.pkl')
            ''',
            "Basic Data Inspection": '''
# View first 5 rows
print(df_csv.head())

# View last 5 rows
print(df_csv.tail())

# Get DataFrame info
print(df_csv.info())

# Summary statistics
print(df_csv.describe())

# Check for missing values
print(df_csv.isnull().sum())

# Display data types
print(df_csv.dtypes)

# Check unique values in a column
print(df_csv['City'].unique())

# Get number of unique values
print(df_csv['City'].nunique())

# Display DataFrame shape
print(df_csv.shape)

# Display DataFrame index
print(df_csv.index)

# Display DataFrame columns
print(df_csv.columns)

# Display DataFrame memory usage
print(df_csv.memory_usage(deep=True))
            '''
        },
        "🔍 Data Exploration": {
            "Selecting Columns": '''
# Select single column as Series
age_series = df_csv['Age']

# Select multiple columns as DataFrame
subset_df = df_csv[['Name', 'Age', 'Salary']]

# Select columns using filter
filtered_df = df_csv.filter(items=['Name', 'Age'])

# Select columns based on data type
numeric_df = df_csv.select_dtypes(include=['int64', 'float64'])

# Select columns using regex
regex_filtered = df_csv.filter(regex='^S', axis=1)  # Select columns starting with 'S'

# Select columns using list comprehension
selected_columns = [col for col in df_csv.columns if 'Sales' in col]
sales_df = df_csv[selected_columns]
            ''',
            "Filtering Rows": '''
# Filter rows where Age > 30
df_over_30 = df_csv[df_csv['Age'] > 30]

# Filter with multiple conditions
df_filtered = df_csv[(df_csv['Age'] > 25) & (df_csv['Salary'] > 50000)]

# Using isin for filtering specific values
df_isin = df_csv[df_csv['City'].isin(['New York', 'Los Angeles'])]

# Filtering rows with string conditions
df_name_contains = df_csv[df_csv['Name'].str.contains('John')]

# Filtering rows using query method
df_query = df_csv.query('Age > 25 and Salary > 50000')

# Filtering rows using loc
df_loc = df_csv.loc[df_csv['Age'] > 25, ['Name', 'Age', 'Salary']]

# Filtering rows using iloc
df_iloc = df_csv.iloc[0:10, 0:3]
            ''',
            "Sorting Data": '''
# Sort by single column ascending
df_sorted = df_csv.sort_values(by='Age')

# Sort by single column descending
df_sorted_desc = df_csv.sort_values(by='Age', ascending=False)

# Sort by multiple columns
df_multi_sorted = df_csv.sort_values(by=['City', 'Age'], ascending=[True, False])

# Sort using inplace
df_csv.sort_values(by='Salary', ascending=True, inplace=True)

# Sort using keys
df_sorted_keys = df_csv.sort_values(by='Salary', key=lambda x: x % 1000)

# Sort index
df_sorted_index = df_csv.sort_index(ascending=True)

# Sort index descending
df_sorted_index_desc = df_csv.sort_index(ascending=False)
            ''',
            "Handling Missing Values": '''
# Drop rows with any missing values
df_dropped = df_csv.dropna()

# Drop rows where specific columns are missing
df_dropped_specific = df_csv.dropna(subset=['Age', 'Salary'])

# Fill missing values with a constant
df_filled = df_csv.fillna(0)

# Fill missing values with mean of the column
df_csv['Age'] = df_csv['Age'].fillna(df_csv['Age'].mean())

# Forward fill
df_ffill = df_csv.fillna(method='ffill')

# Backward fill
df_bfill = df_csv.fillna(method='bfill')

# Fill missing values with interpolation
df_interpolated = df_csv.interpolate(method='linear')

# Fill missing values with different strategies for different columns
df_csv.fillna({'Age': df_csv['Age'].mean(), 'Salary': df_csv['Salary'].median()}, inplace=True)

# Using SimpleImputer from sklearn
imputer = SimpleImputer(strategy='mean')
df_csv[['Age', 'Salary']] = imputer.fit_transform(df_csv[['Age', 'Salary']])
            '''
        },
        "🔄 Data Transformation": {
            "Applying Functions": '''
# Apply a lambda function to a column
df_csv['Age_Plus_One'] = df_csv['Age'].apply(lambda x: x + 1)

# Apply a custom function to a column
def categorize_age(age):
    if age < 18:
        return 'Child'
    elif age < 35:
        return 'Young Adult'
    elif age < 60:
        return 'Adult'
    else:
        return 'Senior'

df_csv['Age_Group'] = df_csv['Age'].apply(categorize_age)

# Apply a function to entire DataFrame
df_cleaned = df_csv.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Apply multiple functions using apply
df_grouped = df_csv.groupby('City').apply(lambda x: x.assign(Age_Squared = x['Age']**2))

# Apply functions with multiple arguments
def add_columns(row, a, b):
    return row['A'] + row['B'] + a + b

df_csv['A_plus_B'] = df_csv.apply(add_columns, args=(5, 10), axis=1)
            ''',
            "Vectorized Operations": '''
# Create new column based on existing columns
df_csv['Salary_Per_Age'] = df_csv['Salary'] / df_csv['Age']

# Vectorized string operations
df_csv['Name_Upper'] = df_csv['Name'].str.upper()
df_csv['City_Lower'] = df_csv['City'].str.lower()

# Boolean operations
df_csv['High_Earner'] = df_csv['Salary'] > 70000

# Creating multiple new columns
df_csv[['Salary_1000s', 'Salary_100s']] = df_csv['Salary'].apply(lambda x: pd.Series([x//1000, x//100]))

# Vectorized arithmetic operations
df_csv['Bonus'] = df_csv['Salary'] * 0.10
df_csv['Total_Compensation'] = df_csv['Salary'] + df_csv['Bonus']

# Vectorized conditional operations using np.where
df_csv['Senior'] = np.where(df_csv['Age'] > 60, 'Yes', 'No')
            ''',
            "Mapping Values": '''
# Map categorical values using a dictionary
city_mapping = {'New York': 'NY', 'Los Angeles': 'LA', 'Chicago': 'CHI'}
df_csv['City_Abbr'] = df_csv['City'].map(city_mapping)

# Replace values directly
df_csv['Gender'].replace({'Male': 'M', 'Female': 'F'}, inplace=True)

# Mapping with function
def map_gender(gender):
    return 'M' if gender == 'Male' else 'F'

df_csv['Gender_Mapped'] = df_csv['Gender'].apply(map_gender)

# Replace multiple values
df_csv['Department'].replace({'HR': 'Human Resources', 'IT': 'Information Technology'}, inplace=True)

# Mapping ordinal categories
priority_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
df_csv['Priority_Level'] = df_csv['Priority'].map(priority_mapping)

# Binning continuous variables
df_csv['Age_Binned'] = pd.cut(df_csv['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])

# Mapping bin labels to numerical codes
age_bin_mapping = {'Child': 1, 'Young Adult': 2, 'Adult': 3, 'Senior': 4}
df_csv['Age_Bin_Code'] = df_csv['Age_Binned'].map(age_bin_mapping)

# Target encoding (mean encoding)
city_salary_mean = df_csv.groupby('City')['Salary'].mean()
df_csv['City_Salary_Mean'] = df_csv['City'].map(city_salary_mean)
            '''
        }
    }

    sections_row2 = {
        "🔗 Merging & Joining": {
            "Merging DataFrames": '''
# Inner join on 'Key'
merged_inner = pd.merge(df1, df2, on='Key', how='inner')

# Left join on 'Key'
merged_left = pd.merge(df1, df2, on='Key', how='left')

# Right join on 'Key'
merged_right = pd.merge(df1, df2, on='Key', how='right')

# Outer join on 'Key'
merged_outer = pd.merge(df1, df2, on='Key', how='outer')

# Merge with indicator
merged_with_indicator = pd.merge(df1, df2, on='Key', how='outer', indicator=True)

# Merge on multiple keys
merged_multiple = pd.merge(df1, df2, on=['Key1', 'Key2'], how='inner')

# Merge with suffixes to handle overlapping column names
merged_suffix = pd.merge(df1, df2, on='Key', how='outer', suffixes=('_left', '_right'))

# Merge with sort
merged_sorted = pd.merge(df1, df2, on='Key', how='inner', sort=True)
            ''',
            "Concatenating DataFrames": '''
# Concatenate vertically (stacking rows)
concatenated_vert = pd.concat([df1, df2, df3], axis=0)

# Concatenate horizontally (stacking columns)
concatenated_horz = pd.concat([df1, df2, df3], axis=1)

# Concatenate with ignore_index
concatenated_ignore = pd.concat([df1, df2], axis=0, ignore_index=True)

# Concatenate along a specific axis with keys
concatenated_keys = pd.concat([df1, df2], axis=0, keys=['Group1', 'Group2'])

# Concatenate with join
concatenated_join = pd.concat([df1, df2], axis=1, join='inner')

# Concatenate multiple DataFrames in a loop
dfs = [df1, df2, df3, df4, df5]
concatenated_loop = pd.concat(dfs, axis=0, ignore_index=True)
            ''',
            "Joining on Multiple Keys": '''
# Merge on multiple keys 'Key1' and 'Key2'
merged_multiple = pd.merge(df1, df2, on=['Key1', 'Key2'], how='inner')

# Joining DataFrames with multiple keys and suffixes
merged_suffix = pd.merge(df1, df2, on=['Key1', 'Key2'], how='outer', suffixes=('_left', '_right'))

# Joining with different join types
merged_inner = pd.merge(df1, df2, on=['Key1', 'Key2'], how='inner')
merged_left = pd.merge(df1, df2, on=['Key1', 'Key2'], how='left')
merged_right = pd.merge(df1, df2, on=['Key1', 'Key2'], how='right')
merged_outer = pd.merge(df1, df2, on=['Key1', 'Key2'], how='outer')
            '''
        },
        "📊 Grouping & Aggregation": {
            "Group By": '''
# Group by single column 'City'
grouped_city = df_csv.groupby('City')

# Group by multiple columns 'City' and 'Age_Group'
grouped_multi = df_csv.groupby(['City', 'Age_Group'])

# Group by with as_index=False to keep grouping columns as columns
grouped_no_index = df_csv.groupby('City', as_index=False)
            ''',
            "Aggregation Functions": '''
# Aggregate with mean
age_mean = grouped_city['Age'].mean()

# Multiple aggregations
aggregated = grouped_city.agg({'Age': ['mean', 'sum'], 'Salary': 'median'})

# Custom aggregation functions
def range_func(x):
    return x.max() - x.min()

custom_agg = grouped_city.agg({
    'Salary': ['mean', 'sum'],
    'Experience': range_func
})

# Aggregating multiple columns with different functions
aggregated_multi = df_csv.groupby('City').agg({
    'Salary': ['mean', 'sum', 'median'],
    'Age': ['min', 'max', 'median'],
    'Experience': ['mean', 'std']
})

# Aggregating with named aggregation
aggregated_named = df_csv.groupby('City').agg(
    Mean_Salary=('Salary', 'mean'),
    Total_Salary=('Salary', 'sum'),
    Median_Age=('Age', 'median')
)
            ''',
            "Pivot Tables": '''
# Simple pivot table
pivot_simple = df_csv.pivot_table(values='Sales', index='Region', columns='Product', aggfunc='sum', fill_value=0)

# Pivot with multiple aggregation functions
pivot_multi = df_csv.pivot_table(values='Sales', index='Region', columns='Product', aggfunc=['sum', 'mean'], fill_value=0)

# Pivot with margins (totals)
pivot_margins = df_csv.pivot_table(values='Sales', index='Region', columns='Product', aggfunc='sum', margins=True, fill_value=0)

# Pivot with custom aggregation functions
pivot_custom = df_csv.pivot_table(values='Sales', index='Region', columns='Product', aggfunc={'Sales': ['sum', 'mean']}, fill_value=0)

# Pivot with multiple indexes
pivot_multi_index = df_csv.pivot_table(values='Sales', index=['Region', 'City'], columns='Product', aggfunc='sum', fill_value=0)

# Pivot with multiple values
pivot_multi_values = df_csv.pivot_table(values=['Sales', 'Profit'], index='Region', columns='Product', aggfunc='sum', fill_value=0)
            '''
        },
        "📈 Data Visualization": {
            "Plotting with Pandas": '''
# Line plot for Sales over Time
df_csv.plot(kind='line', x='Date', y='Sales', title='Sales Over Time', figsize=(10,6), color='blue', marker='o')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# Bar plot for Sales by City
df_csv.plot(kind='bar', x='City', y='Sales', title='Sales by City', figsize=(10,6), color='skyblue')
plt.xlabel('City')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter plot for Age vs Salary
df_csv.plot(kind='scatter', x='Age', y='Salary', title='Age vs Salary', figsize=(10,6), color='red', alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Salary')
plt.grid(True)
plt.show()

# Histogram for Age Distribution
df_csv['Age'].plot(kind='hist', bins=10, title='Age Distribution', figsize=(10,6), color='green', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Boxplot for Salary Distribution by City
df_csv.boxplot(column='Salary', by='City', title='Salary Distribution by City', figsize=(10,6))
plt.xlabel('City')
plt.ylabel('Salary')
plt.show()
            ''',
            "Advanced Visualization": '''
# Heatmap using seaborn
plt.figure(figsize=(10,8))
sns.heatmap(df_csv.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Pairplot using seaborn
sns.pairplot(df_csv, hue='City', markers=["o", "s", "D"], palette='Set2')
plt.suptitle('Pairplot of DataFrame', y=1.02)
plt.show()

# Boxplot using seaborn
plt.figure(figsize=(10,6))
sns.boxplot(x='City', y='Salary', data=df_csv, palette='Pastel1')
plt.title('Salary Distribution by City')
plt.show()

# Violin plot using seaborn
plt.figure(figsize=(10,6))
sns.violinplot(x='City', y='Salary', data=df_csv, palette='Set3')
plt.title('Salary Distribution by City')
plt.show()

# Swarm plot using seaborn
plt.figure(figsize=(10,6))
sns.swarmplot(x='City', y='Salary', data=df_csv, hue='Gender', palette='Set1')
plt.title('Salary Distribution by City and Gender')
plt.legend(title='Gender')
plt.show()

# Jointplot using seaborn
sns.jointplot(x='Age', y='Salary', data=df_csv, kind='scatter', color='green')
plt.show()

# KDE plot using seaborn
sns.kdeplot(data=df_csv, x='Age', y='Salary', cmap='Blues', shade=True)
plt.title('KDE Plot of Age vs Salary')
plt.show()

# FacetGrid using seaborn
g = sns.FacetGrid(df_csv, col='City', hue='Gender')
g.map(plt.scatter, 'Age', 'Salary').add_legend()
plt.show()

# Countplot using seaborn
sns.countplot(x='City', data=df_csv, palette='Set2')
plt.title('Count of Records by City')
plt.show()

# Stripplot using seaborn
sns.stripplot(x='City', y='Salary', data=df_csv, jitter=True, hue='Gender', dodge=True)
plt.title('Stripplot of Salary by City and Gender')
plt.legend(title='Gender')
plt.show()

# Rugplot using seaborn
sns.rugplot(x='Salary', data=df_csv, height=0.05)
plt.title('Rugplot of Salary')
plt.show()

# Heatmap of categorical data
category_counts = df_csv.groupby(['City', 'Gender']).size().unstack()
sns.heatmap(category_counts, annot=True, fmt="d", cmap='YlGnBu')
plt.title('Heatmap of City vs Gender Counts')
plt.show()
            '''
        },
        "🔧 Data Engineering": {
            "Exporting Data": '''
# Export DataFrame to CSV
df_csv.to_csv('output.csv', index=False)

# Export DataFrame to Excel with specific sheet name
df_csv.to_excel('output.xlsx', index=False, sheet_name='DataSheet')

# Export DataFrame to JSON
df_csv.to_json('output.json', orient='records', lines=True)

# Export DataFrame to SQL database
engine = create_engine('postgresql://user:password@localhost:5432/mydatabase')
df_csv.to_sql('table_name', engine, if_exists='replace', index=False)

# Export DataFrame to HDF5
df_csv.to_hdf('output.h5', key='df_key', mode='w')

# Export DataFrame to Parquet
df_csv.to_parquet('output.parquet')

# Export DataFrame to Pickle
df_csv.to_pickle('output.pkl')

# Export DataFrame to Feather
df_csv.to_feather('output.feather')

# Export DataFrame to Msgpack (Deprecated in newer Pandas versions)
# df_csv.to_msgpack('output.msgpack')  # Not recommended
            ''',
            "Handling Large Datasets": '''
# Read large CSV in chunks
chunksize = 10**6
chunks = []
for chunk in pd.read_csv('large_data.csv', chunksize=chunksize):
    # Process each chunk
    processed_chunk = chunk[chunk['Age'] > 30]
    chunks.append(processed_chunk)
df_large = pd.concat(chunks, axis=0)

# Optimize memory usage by changing data types
df_csv['Age'] = df_csv['Age'].astype('int16')
df_csv['Salary'] = df_csv['Salary'].astype('float32')
df_csv['Gender'] = df_csv['Gender'].astype('category')

# Reduce memory usage by converting object types to category
for col in df_csv.select_dtypes(include=['object']).columns:
    df_csv[col] = df_csv[col].astype('category')

# Using memory-efficient data types
df_csv['ZipCode'] = df_csv['ZipCode'].astype('int32')
df_csv['Income'] = df_csv['Income'].astype('float32')

# Dropping unnecessary columns to save memory
df_csv.drop(['UnnecessaryColumn1', 'UnnecessaryColumn2'], axis=1, inplace=True)

# Sampling large datasets for quick analysis
df_sample = df_csv.sample(frac=0.1, random_state=1)

# Using Dask for parallel processing (if dataset is extremely large)
# import dask.dataframe as dd
# ddf = dd.read_csv('very_large_data.csv')
# ddf_filtered = ddf[ddf['Age'] > 30]
# df_filtered = ddf_filtered.compute()
            ''',
            "Using SQL with Pandas": '''
# Querying SQL database and loading into DataFrame
query = """
SELECT Name, Age, Salary
FROM employees
WHERE Age > 25
ORDER BY Salary DESC
"""
df_sql = pd.read_sql(query, engine)

# Display the DataFrame
print(df_sql.head())

# Insert data into SQL database from DataFrame
df_new = pd.DataFrame({
    'Name': ['Alice', 'Bob'],
    'Age': [28, 34],
    'Salary': [70000, 80000]
})
df_new.to_sql('employees', engine, if_exists='append', index=False)

# Update records in SQL database
update_query = """
UPDATE employees
SET Salary = Salary * 1.10
WHERE Age > 30
"""
with engine.connect() as conn:
    conn.execute(update_query)

# Delete records from SQL database
delete_query = """
DELETE FROM employees
WHERE Name = 'Bob'
"""
with engine.connect() as conn:
    conn.execute(delete_query)

# Read from SQL database into DataFrame with parameters
from sqlalchemy import text
sql_query = text("SELECT * FROM employees WHERE Salary > :salary_threshold")
df_sql_param = pd.read_sql(sql_query, engine, params={"salary_threshold": 75000})

# Using SQLAlchemy ORM for more complex queries
from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)
session = Session()

# Example ORM query
result = session.execute("SELECT * FROM employees WHERE Age > 30").fetchall()
df_orm = pd.DataFrame(result, columns=result[0].keys())

# Closing the session
session.close()
            '''
        },
        "🔍 Advanced Topics": {
            "Time Series Analysis": '''
# Convert 'Date' column to datetime
df_csv['Date'] = pd.to_datetime(df_csv['Date'])

# Set 'Date' as index
df_csv.set_index('Date', inplace=True)

# Resample data to monthly frequency and sum Sales
monthly_sales = df_csv['Sales'].resample('M').sum()

# Plot resampled data
monthly_sales.plot(kind='line', title='Sales Over Time', figsize=(12,6), color='purple', marker='x')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# Rolling statistics: 12-month moving average
df_csv['Sales_MA12'] = df_csv['Sales'].rolling(window=12).mean()
df_csv['Sales_MA12'].plot(title='12-Month Moving Average of Sales', figsize=(12,6), color='orange')
plt.xlabel('Date')
plt.ylabel('Sales (MA12)')
plt.grid(True)
plt.show()

# Decompose time series
decomposition = seasonal_decompose(df_csv['Sales'], model='additive')
fig = decomposition.plot()
plt.show()

# Stationarity tests
from statsmodels.tsa.stattools import adfuller

result = adfuller(df_csv['Sales'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Differencing to make series stationary
df_csv['Sales_Diff'] = df_csv['Sales'].diff()
df_csv['Sales_Diff'].dropna().plot(title='Differenced Sales', figsize=(12,6))
plt.xlabel('Date')
plt.ylabel('Sales Difference')
plt.show()

# Autocorrelation
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df_csv['Sales'].dropna(), lags=40)
plt.title('Autocorrelation of Sales')
plt.show()

plot_pacf(df_csv['Sales'].dropna(), lags=40)
plt.title('Partial Autocorrelation of Sales')
plt.show()

# Forecasting with ARIMA (example)
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df_csv['Sales'], order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

# Forecasting future values
forecast = model_fit.forecast(steps=12)
print(forecast)
forecast.plot(title='Sales Forecast', figsize=(12,6))
plt.xlabel('Date')
plt.ylabel('Forecasted Sales')
plt.show()

# Seasonal decomposition using STL
from statsmodels.tsa.seasonal import STL

stl = STL(df_csv['Sales'], seasonal=13)
result = stl.fit()
result.plot()
plt.show()

# Trend extraction
df_csv['Trend'] = result.trend
df_csv['Trend'].plot(title='Trend Component', figsize=(12,6))
plt.xlabel('Date')
plt.ylabel('Trend')
plt.show()

# Seasonal extraction
df_csv['Seasonal'] = result.seasonal
df_csv['Seasonal'].plot(title='Seasonal Component', figsize=(12,6))
plt.xlabel('Date')
plt.ylabel('Seasonality')
plt.show()

# Residual extraction
df_csv['Residual'] = result.resid
df_csv['Residual'].plot(title='Residual Component', figsize=(12,6))
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.show()

# Forecast with confidence intervals
forecast_ci = model_fit.get_forecast(steps=12)
forecast_df = forecast_ci.summary_frame()
print(forecast_df)
forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']].plot(figsize=(12,6))
plt.title('Sales Forecast with Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()
            ''',
            "Handling Categorical Data": '''
# One-Hot Encoding using get_dummies
df_encoded = pd.get_dummies(df_csv, columns=['Category'], drop_first=True)

# Label Encoding using sklearn
le = LabelEncoder()
df_csv['Category_Encoded'] = le.fit_transform(df_csv['Category'])

# Handling ordinal categories
ordinal_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
df_csv['Priority_Level'] = df_csv['Priority'].map(ordinal_mapping)

# Creating dummy variables for multiple categorical columns
df_dummies = pd.get_dummies(df_csv, columns=['City', 'Gender'], drop_first=True)

# Encoding ordinal categories with pandas.Categorical
df_csv['Quality'] = pd.Categorical(df_csv['Quality'], categories=['Poor', 'Average', 'Good', 'Excellent'], ordered=True)
df_csv['Quality_Code'] = df_csv['Quality'].cat.codes

# Binning continuous variables
df_csv['Age_Binned'] = pd.cut(df_csv['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])

# Mapping bin labels to numerical codes
age_bin_mapping = {'Child': 1, 'Young Adult': 2, 'Adult': 3, 'Senior': 4}
df_csv['Age_Bin_Code'] = df_csv['Age_Binned'].map(age_bin_mapping)

# Target encoding (mean encoding)
city_salary_mean = df_csv.groupby('City')['Salary'].mean()
df_csv['City_Salary_Mean'] = df_csv['City'].map(city_salary_mean)
            ''',
            "Applying Functions with GroupBy": '''
# Define a custom aggregation function
def range_func(x):
    return x.max() - x.min()

# Apply custom function to 'Experience' column
experience_range = df_csv.groupby('City')['Experience'].apply(range_func)
print(experience_range)

# Apply multiple aggregation functions
aggregated = df_csv.groupby('City').agg({
    'Salary': ['mean', 'sum', 'median'],
    'Age': 'max',
    'Experience': range_func
})
print(aggregated)

# Applying functions to multiple columns
aggregated_multi = df_csv.groupby('City').agg({
    'Salary': ['mean', 'sum', 'median'],
    'Age': ['min', 'max', 'median'],
    'Experience': ['mean', 'std']
})
print(aggregated_multi)

# Using transform to broadcast group results
df_csv['Salary_Mean'] = df_csv.groupby('City')['Salary'].transform('mean')
print(df_csv.head())

# Aggregating with multiple custom functions
def custom_sum(x):
    return x.sum()

def custom_mean(x):
    return x.mean()

aggregated_custom = df_csv.groupby('City').agg({
    'Salary': [custom_sum, custom_mean],
    'Age': 'mean'
})
print(aggregated_custom)

# Aggregating with named aggregation
aggregated_named = df_csv.groupby('City').agg(
    Mean_Salary=('Salary', 'mean'),
    Total_Salary=('Salary', 'sum'),
    Median_Age=('Age', 'median')
)
print(aggregated_named)
            '''
        },
        "📈 Data Visualization": {
            "Plotting with Pandas": '''
# Line plot for Sales over Time
df_csv.plot(kind='line', x='Date', y='Sales', title='Sales Over Time', figsize=(10,6), color='blue', marker='o')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# Bar plot for Sales by City
df_csv.plot(kind='bar', x='City', y='Sales', title='Sales by City', figsize=(10,6), color='skyblue')
plt.xlabel('City')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter plot for Age vs Salary
df_csv.plot(kind='scatter', x='Age', y='Salary', title='Age vs Salary', figsize=(10,6), color='red', alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Salary')
plt.grid(True)
plt.show()

# Histogram for Age Distribution
df_csv['Age'].plot(kind='hist', bins=10, title='Age Distribution', figsize=(10,6), color='green', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Boxplot for Salary Distribution by City
df_csv.boxplot(column='Salary', by='City', title='Salary Distribution by City', figsize=(10,6))
plt.xlabel('City')
plt.ylabel('Salary')
plt.show()
            ''',
            "Advanced Visualization": '''
# Heatmap using seaborn
plt.figure(figsize=(10,8))
sns.heatmap(df_csv.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Pairplot using seaborn
sns.pairplot(df_csv, hue='City', markers=["o", "s", "D"], palette='Set2')
plt.suptitle('Pairplot of DataFrame', y=1.02)
plt.show()

# Boxplot using seaborn
plt.figure(figsize=(10,6))
sns.boxplot(x='City', y='Salary', data=df_csv, palette='Pastel1')
plt.title('Salary Distribution by City')
plt.show()

# Violin plot using seaborn
plt.figure(figsize=(10,6))
sns.violinplot(x='City', y='Salary', data=df_csv, palette='Set3')
plt.title('Salary Distribution by City')
plt.show()

# Swarm plot using seaborn
plt.figure(figsize=(10,6))
sns.swarmplot(x='City', y='Salary', data=df_csv, hue='Gender', palette='Set1')
plt.title('Salary Distribution by City and Gender')
plt.legend(title='Gender')
plt.show()

# Jointplot using seaborn
sns.jointplot(x='Age', y='Salary', data=df_csv, kind='scatter', color='green')
plt.show()

# KDE plot using seaborn
sns.kdeplot(data=df_csv, x='Age', y='Salary', cmap='Blues', shade=True)
plt.title('KDE Plot of Age vs Salary')
plt.show()

# FacetGrid using seaborn
g = sns.FacetGrid(df_csv, col='City', hue='Gender')
g.map(plt.scatter, 'Age', 'Salary').add_legend()
plt.show()

# Countplot using seaborn
sns.countplot(x='City', data=df_csv, palette='Set2')
plt.title('Count of Records by City')
plt.show()

# Stripplot using seaborn
sns.stripplot(x='City', y='Salary', data=df_csv, jitter=True, hue='Gender', dodge=True)
plt.title('Stripplot of Salary by City and Gender')
plt.legend(title='Gender')
plt.show()

# Rugplot using seaborn
sns.rugplot(x='Salary', data=df_csv, height=0.05)
plt.title('Rugplot of Salary')
plt.show()

# Heatmap of categorical data
category_counts = df_csv.groupby(['City', 'Gender']).size().unstack()
sns.heatmap(category_counts, annot=True, fmt="d", cmap='YlGnBu')
plt.title('Heatmap of City vs Gender Counts')
plt.show()
            '''
        },
        "🔧 Data Engineering": {
            "Exporting Data": '''
# Export DataFrame to CSV
df_csv.to_csv('output.csv', index=False)

# Export DataFrame to Excel with specific sheet name
df_csv.to_excel('output.xlsx', index=False, sheet_name='DataSheet')

# Export DataFrame to JSON
df_csv.to_json('output.json', orient='records', lines=True)

# Export DataFrame to SQL database
engine = create_engine('postgresql://user:password@localhost:5432/mydatabase')
df_csv.to_sql('table_name', engine, if_exists='replace', index=False)

# Export DataFrame to HDF5
df_csv.to_hdf('output.h5', key='df_key', mode='w')

# Export DataFrame to Parquet
df_csv.to_parquet('output.parquet')

# Export DataFrame to Pickle
df_csv.to_pickle('output.pkl')

# Export DataFrame to Feather
df_csv.to_feather('output.feather')

# Export DataFrame to Msgpack (Deprecated in newer Pandas versions)
# df_csv.to_msgpack('output.msgpack')  # Not recommended
            ''',
            "Handling Large Datasets": '''
# Read large CSV in chunks
chunksize = 10**6
chunks = []
for chunk in pd.read_csv('large_data.csv', chunksize=chunksize):
    # Process each chunk
    processed_chunk = chunk[chunk['Age'] > 30]
    chunks.append(processed_chunk)
df_large = pd.concat(chunks, axis=0)

# Optimize memory usage by changing data types
df_csv['Age'] = df_csv['Age'].astype('int16')
df_csv['Salary'] = df_csv['Salary'].astype('float32')
df_csv['Gender'] = df_csv['Gender'].astype('category')

# Reduce memory usage by converting object types to category
for col in df_csv.select_dtypes(include=['object']).columns:
    df_csv[col] = df_csv[col].astype('category')

# Using memory-efficient data types
df_csv['ZipCode'] = df_csv['ZipCode'].astype('int32')
df_csv['Income'] = df_csv['Income'].astype('float32')

# Dropping unnecessary columns to save memory
df_csv.drop(['UnnecessaryColumn1', 'UnnecessaryColumn2'], axis=1, inplace=True)

# Sampling large datasets for quick analysis
df_sample = df_csv.sample(frac=0.1, random_state=1)

# Using Dask for parallel processing (if dataset is extremely large)
# import dask.dataframe as dd
# ddf = dd.read_csv('very_large_data.csv')
# ddf_filtered = ddf[ddf['Age'] > 30]
# df_filtered = ddf_filtered.compute()
            ''',
            "Using SQL with Pandas": '''
# Querying SQL database and loading into DataFrame
query = """
SELECT Name, Age, Salary
FROM employees
WHERE Age > 25
ORDER BY Salary DESC
"""
df_sql = pd.read_sql(query, engine)

# Display the DataFrame
print(df_sql.head())

# Insert data into SQL database from DataFrame
df_new = pd.DataFrame({
    'Name': ['Alice', 'Bob'],
    'Age': [28, 34],
    'Salary': [70000, 80000]
})
df_new.to_sql('employees', engine, if_exists='append', index=False)

# Update records in SQL database
update_query = """
UPDATE employees
SET Salary = Salary * 1.10
WHERE Age > 30
"""
with engine.connect() as conn:
    conn.execute(update_query)

# Delete records from SQL database
delete_query = """
DELETE FROM employees
WHERE Name = 'Bob'
"""
with engine.connect() as conn:
    conn.execute(delete_query)

# Read from SQL database into DataFrame with parameters
from sqlalchemy import text
sql_query = text("SELECT * FROM employees WHERE Salary > :salary_threshold")
df_sql_param = pd.read_sql(sql_query, engine, params={"salary_threshold": 75000})

# Using SQLAlchemy ORM for more complex queries
from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)
session = Session()

# Example ORM query
result = session.execute("SELECT * FROM employees WHERE Age > 30").fetchall()
df_orm = pd.DataFrame(result, columns=result[0].keys())

# Closing the session
session.close()
            '''
        },
        "🔍 Advanced Topics": {
            "Time Series Analysis": '''
# Convert 'Date' column to datetime
df_csv['Date'] = pd.to_datetime(df_csv['Date'])

# Set 'Date' as index
df_csv.set_index('Date', inplace=True)

# Resample data to monthly frequency and sum Sales
monthly_sales = df_csv['Sales'].resample('M').sum()

# Plot resampled data
monthly_sales.plot(kind='line', title='Sales Over Time', figsize=(12,6), color='purple', marker='x')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# Rolling statistics: 12-month moving average
df_csv['Sales_MA12'] = df_csv['Sales'].rolling(window=12).mean()
df_csv['Sales_MA12'].plot(title='12-Month Moving Average of Sales', figsize=(12,6), color='orange')
plt.xlabel('Date')
plt.ylabel('Sales (MA12)')
plt.grid(True)
plt.show()

# Decompose time series
decomposition = seasonal_decompose(df_csv['Sales'], model='additive')
fig = decomposition.plot()
plt.show()

# Stationarity tests
from statsmodels.tsa.stattools import adfuller

result = adfuller(df_csv['Sales'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Differencing to make series stationary
df_csv['Sales_Diff'] = df_csv['Sales'].diff()
df_csv['Sales_Diff'].dropna().plot(title='Differenced Sales', figsize=(12,6))
plt.xlabel('Date')
plt.ylabel('Sales Difference')
plt.show()

# Autocorrelation
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df_csv['Sales'].dropna(), lags=40)
plt.title('Autocorrelation of Sales')
plt.show()

plot_pacf(df_csv['Sales'].dropna(), lags=40)
plt.title('Partial Autocorrelation of Sales')
plt.show()

# Forecasting with ARIMA (example)
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df_csv['Sales'], order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

# Forecasting future values
forecast = model_fit.forecast(steps=12)
print(forecast)
forecast.plot(title='Sales Forecast', figsize=(12,6))
plt.xlabel('Date')
plt.ylabel('Forecasted Sales')
plt.show()

# Seasonal decomposition using STL
from statsmodels.tsa.seasonal import STL

stl = STL(df_csv['Sales'], seasonal=13)
result = stl.fit()
result.plot()
plt.show()

# Trend extraction
df_csv['Trend'] = result.trend
df_csv['Trend'].plot(title='Trend Component', figsize=(12,6))
plt.xlabel('Date')
plt.ylabel('Trend')
plt.show()

# Seasonal extraction
df_csv['Seasonal'] = result.seasonal
df_csv['Seasonal'].plot(title='Seasonal Component', figsize=(12,6))
plt.xlabel('Date')
plt.ylabel('Seasonality')
plt.show()

# Residual extraction
df_csv['Residual'] = result.resid
df_csv['Residual'].plot(title='Residual Component', figsize=(12,6))
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.show()

# Forecast with confidence intervals
forecast_ci = model_fit.get_forecast(steps=12)
forecast_df = forecast_ci.summary_frame()
print(forecast_df)
forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']].plot(figsize=(12,6))
plt.title('Sales Forecast with Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()
            ''',
            "Handling Categorical Data": '''
# One-Hot Encoding using get_dummies
df_encoded = pd.get_dummies(df_csv, columns=['Category'], drop_first=True)

# Label Encoding using sklearn
le = LabelEncoder()
df_csv['Category_Encoded'] = le.fit_transform(df_csv['Category'])

# Handling ordinal categories
ordinal_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
df_csv['Priority_Level'] = df_csv['Priority'].map(ordinal_mapping)

# Creating dummy variables for multiple categorical columns
df_dummies = pd.get_dummies(df_csv, columns=['City', 'Gender'], drop_first=True)

# Encoding ordinal categories with pandas.Categorical
df_csv['Quality'] = pd.Categorical(df_csv['Quality'], categories=['Poor', 'Average', 'Good', 'Excellent'], ordered=True)
df_csv['Quality_Code'] = df_csv['Quality'].cat.codes

# Binning continuous variables
df_csv['Age_Binned'] = pd.cut(df_csv['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])

# Mapping bin labels to numerical codes
age_bin_mapping = {'Child': 1, 'Young Adult': 2, 'Adult': 3, 'Senior': 4}
df_csv['Age_Bin_Code'] = df_csv['Age_Binned'].map(age_bin_mapping)

# Target encoding (mean encoding)
city_salary_mean = df_csv.groupby('City')['Salary'].mean()
df_csv['City_Salary_Mean'] = df_csv['City'].map(city_salary_mean)
            ''',
            "Applying Functions with GroupBy": '''
# Define a custom aggregation function
def range_func(x):
    return x.max() - x.min()

# Apply custom function to 'Experience' column
experience_range = df_csv.groupby('City')['Experience'].apply(range_func)
print(experience_range)

# Apply multiple aggregation functions
aggregated = df_csv.groupby('City').agg({
    'Salary': ['mean', 'sum', 'median'],
    'Age': 'max',
    'Experience': range_func
})
print(aggregated)

# Applying functions to multiple columns
aggregated_multi = df_csv.groupby('City').agg({
    'Salary': ['mean', 'sum', 'median'],
    'Age': ['min', 'max', 'median'],
    'Experience': ['mean', 'std']
})
print(aggregated_multi)

# Using transform to broadcast group results
df_csv['Salary_Mean'] = df_csv.groupby('City')['Salary'].transform('mean')
print(df_csv.head())

# Aggregating with multiple custom functions
def custom_sum(x):
    return x.sum()

def custom_mean(x):
    return x.mean()

aggregated_custom = df_csv.groupby('City').agg({
    'Salary': [custom_sum, custom_mean],
    'Age': 'mean'
})
print(aggregated_custom)

# Aggregating with named aggregation
aggregated_named = df_csv.groupby('City').agg(
    Mean_Salary=('Salary', 'mean'),
    Total_Salary=('Salary', 'sum'),
    Median_Age=('Age', 'median')
)
print(aggregated_named)
            '''
        }
    }

    # Split main categories into two rows
    main_categories_row1 = list(sections_row1.keys())
    main_categories_row2 = list(sections_row2.keys())

    # Function to render a row of tabs without hiding content
    def render_tab_row(categories, sections):
        # Create tabs for the given categories
        tabs = st.tabs(categories)
        for tab, category in zip(tabs, categories):
            with tab:
                # Optional: Add section animation
                if lottie_section:
                    st_lottie(lottie_section, height=100, key=f"{category}_animation")
                # Iterate through subtopics and display code snippets
                for sub_title, code in sections[category].items():
                    st.markdown(f"### {sub_title}")
                    st.code(code, language='python')
                    st.markdown("<br>", unsafe_allow_html=True)

    # Layout using containers for each row
    with st.container():
        st.markdown("<h2 style='text-align: center; color: #333333;'>📚 Foundational Pandas Topics</h2>", unsafe_allow_html=True)
        render_tab_row(main_categories_row1, sections_row1)

    with st.container():
        st.markdown("<h2 style='text-align: center; color: #333333;'>🚀 Advanced Pandas Topics</h2>", unsafe_allow_html=True)
        render_tab_row(main_categories_row2, sections_row2)

    # Footer with social media links and animation
    st.markdown(f"""
        <div style="background-color: #FFFFFF; color: black; text-align: center; padding: 20px; margin-top: 50px; border-top: 2px solid #000000;">
            <p>Connect with me:</p>
            <div style="display: flex; justify-content: center; gap: 20px;">
                <a href="https://facebook.com/ahammadmejbah" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/733/733547.png" alt="Facebook" width="30" style="transition: transform 0.2s;">
                </a>
                <a href="https://instagram.com/ahammadmejbah" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/733/733558.png" alt="Instagram" width="30" style="transition: transform 0.2s;">
                </a>
                <a href="https://github.com/ahammadmejbah" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" alt="GitHub" width="30" style="transition: transform 0.2s;">
                </a>
                <a href="https://ahammadmejbah.com/" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/919/919827.png" alt="Portfolio" width="30" style="transition: transform 0.2s;">
                </a>
            </div>
            <br>
            <small>Pandas Cheat Sheet v1.0 | Nov 2024 | <a href="https://ahammadmejbah.com/" style="color: #333333;">Mejbah Ahammad</a></small>
            <div class="card-footer">Mejbah Ahammad © 2024</div>
        </div>
    """, unsafe_allow_html=True)

    # Optional: Add some spacing at the bottom
    st.markdown("<br><br><br>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
