import streamlit as st
import base64
import requests
from streamlit_lottie import st_lottie
import json

# Initial page configuration
st.set_page_config(
    page_title='📊 Pandas Cheat Sheet By Mejbah Ahammad',
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

# Sidebar content
def ds_sidebar():
    logo_url = 'https://ahammadmejbah.com/content/images/2024/10/Mejbah-Ahammad-Profile-8.png'
    logo_encoded = img_to_bytes(logo_url)
    
    st.sidebar.markdown(
        f"""
        <a href="https://ahammadmejbah.com/">
            <img src='data:image/png;base64,{logo_encoded}' class='img-fluid' width=100>
        </a>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.header('🧰 Pandas Cheat Sheet')
    
    st.sidebar.markdown('''
    <small>Comprehensive summary of essential Pandas concepts, functions, and best practices.</small>
    ''', unsafe_allow_html=True)
    
    st.sidebar.markdown('__🔑 Key Libraries__')
    st.sidebar.code('''
$ pip install pandas numpy matplotlib seaborn
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
    ''')
    
    st.sidebar.markdown('''<hr>''', unsafe_allow_html=True)
    st.sidebar.markdown('''<small>[Pandas Cheat Sheet v1.0](https://github.com/ahammadmejbah/Data-Science-Cheat-Sheet) | Nov 2024 | [Mejbah Ahammad](https://ahammadmejbah.com/)<div class="card-footer">Mejbah Ahammad © 2024</div></small>''', unsafe_allow_html=True)

# Main body of cheat sheet
def ds_body():
    # Load Lottie animations
    lottie_header = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_ydo1amjm.json")  # Replace with your preferred animation
    lottie_section = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_1pxqjqps.json")  # Replace with your preferred animation
    
    # Header with animation
    st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #FF4B4B;">📊 Pandas Cheat Sheet By Mejbah Ahammad</h1>
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

# Setting display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)
            ''',
            "Reading Data": '''
# Read CSV with specific encoding
df = pd.read_csv('data.csv', encoding='utf-8')

# Read Excel specifying sheet
df = pd.read_excel('data.xlsx', sheet_name='Sheet1', engine='openpyxl')

# Read JSON with normalization
df = pd.read_json('data.json')
df_normalized = pd.json_normalize(json_data, 'records')

# Read from SQL database
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:password@localhost:5432/mydatabase')
df = pd.read_sql('SELECT * FROM table_name WHERE age > 30', engine)
            ''',
            "Basic Data Inspection": '''
# View first 5 rows
print(df.head())

# View last 5 rows
print(df.tail())

# Get DataFrame info
print(df.info())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Display data types
print(df.dtypes)
            '''
        },
        "🔍 Data Exploration": {
            "Selecting Columns": '''
# Select single column
age_series = df['Age']

# Select multiple columns
subset_df = df[['Name', 'Age', 'Salary']]

# Select columns using filter
filtered_df = df.filter(items=['Name', 'Age'])
            ''',
            "Filtering Rows": '''
# Filter rows where Age > 30
df_over_30 = df[df['Age'] > 30]

# Filter with multiple conditions
df_filtered = df[(df['Age'] > 25) & (df['Salary'] > 50000)]

# Using isin for filtering
df_isin = df[df['City'].isin(['New York', 'Los Angeles'])]
            ''',
            "Sorting Data": '''
# Sort by single column ascending
df_sorted = df.sort_values(by='Age')

# Sort by single column descending
df_sorted_desc = df.sort_values(by='Age', ascending=False)

# Sort by multiple columns
df_multi_sorted = df.sort_values(by=['City', 'Age'], ascending=[True, False])
            ''',
            "Handling Missing Values": '''
# Drop rows with any missing values
df_dropped = df.dropna()

# Drop rows where specific columns are missing
df_dropped_specific = df.dropna(subset=['Age', 'Salary'])

# Fill missing values with a constant
df_filled = df.fillna(0)

# Fill missing values with mean of the column
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Forward fill
df_ffill = df.fillna(method='ffill')

# Backward fill
df_bfill = df.fillna(method='bfill')
            '''
        },
        "🔄 Data Transformation": {
            "Applying Functions": '''
# Apply a lambda function to a column
df['Age_Plus_One'] = df['Age'].apply(lambda x: x + 1)

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

df['Age_Group'] = df['Age'].apply(categorize_age)

# Apply a function to entire DataFrame
df_cleaned = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            ''',
            "Vectorized Operations": '''
# Create new column based on existing columns
df['Salary_Per_Age'] = df['Salary'] / df['Age']

# Vectorized string operations
df['Name_Upper'] = df['Name'].str.upper()
df['City_Lower'] = df['City'].str.lower()

# Boolean operations
df['High_Earner'] = df['Salary'] > 70000
            ''',
            "Mapping Values": '''
# Map categorical values using a dictionary
city_mapping = {'New York': 'NY', 'Los Angeles': 'LA', 'Chicago': 'CHI'}
df['City_Abbr'] = df['City'].map(city_mapping)

# Replace values directly
df['Gender'].replace({'Male': 'M', 'Female': 'F'}, inplace=True)
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
            ''',
            "Concatenating DataFrames": '''
# Concatenate vertically (stacking rows)
concatenated_vert = pd.concat([df1, df2, df3], axis=0)

# Concatenate horizontally (stacking columns)
concatenated_horz = pd.concat([df1, df2, df3], axis=1)
            ''',
            "Joining on Multiple Keys": '''
# Merge on multiple keys 'Key1' and 'Key2'
merged_multiple = pd.merge(df1, df2, on=['Key1', 'Key2'], how='inner')
            '''
        },
        "📊 Grouping & Aggregation": {
            "Group By": '''
# Group by single column 'City'
grouped_city = df.groupby('City')

# Group by multiple columns 'City' and 'Age_Group'
grouped_multi = df.groupby(['City', 'Age_Group'])
            ''',
            "Aggregation Functions": '''
# Aggregate with mean
age_mean = grouped_city['Age'].mean()

# Multiple aggregations
aggregated = grouped_city.agg({'Age': ['mean', 'sum'], 'Salary': 'median'})

# Custom aggregation functions
custom_agg = grouped_city.agg({
    'Salary': ['mean', 'sum'],
    'Experience': lambda x: x.max() - x.min()
})
            ''',
            "Pivot Tables": '''
# Simple pivot table
pivot_simple = df.pivot_table(values='Sales', index='Region', columns='Product', aggfunc='sum', fill_value=0)

# Pivot with multiple aggregation functions
pivot_multi = df.pivot_table(values='Sales', index='Region', columns='Product', aggfunc=['sum', 'mean'], fill_value=0)

# Pivot with margins (totals)
pivot_margins = df.pivot_table(values='Sales', index='Region', columns='Product', aggfunc='sum', margins=True, fill_value=0)
            '''
        },
        "📈 Data Visualization": {
            "Plotting with Pandas": '''
# Line plot for Sales over Time
df.plot(kind='line', x='Date', y='Sales', title='Sales Over Time', figsize=(10,6))

# Bar plot for Sales by City
df.plot(kind='bar', x='City', y='Sales', title='Sales by City', figsize=(10,6), color='skyblue')

# Scatter plot for Age vs Salary
df.plot(kind='scatter', x='Age', y='Salary', title='Age vs Salary', figsize=(10,6), color='red')

# Histogram for Age Distribution
df['Age'].plot(kind='hist', bins=10, title='Age Distribution', figsize=(10,6), color='green', edgecolor='black')

# Boxplot for Salary Distribution by City
df.boxplot(column='Salary', by='City', title='Salary Distribution by City', figsize=(10,6))
            ''',
            "Advanced Visualization": '''
# Heatmap using seaborn
import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot using seaborn
sns.pairplot(df, hue='City', markers=["o", "s", "D"], palette='Set2')
plt.title('Pairplot of DataFrame')
plt.show()

# Boxplot using seaborn
plt.figure(figsize=(10,6))
sns.boxplot(x='City', y='Salary', data=df)
plt.title('Salary Distribution by City')
plt.show()

# Violin plot using seaborn
plt.figure(figsize=(10,6))
sns.violinplot(x='City', y='Salary', data=df, palette='Pastel1')
plt.title('Salary Distribution by City')
plt.show()
            '''
        },
        "🔧 Data Engineering": {
            "Exporting Data": '''
# Export DataFrame to CSV
df.to_csv('output.csv', index=False)

# Export DataFrame to Excel with specific sheet name
df.to_excel('output.xlsx', index=False, sheet_name='DataSheet')

# Export DataFrame to JSON
df.to_json('output.json', orient='records', lines=True)

# Export DataFrame to SQL database
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:password@localhost:5432/mydatabase')
df.to_sql('table_name', engine, if_exists='replace', index=False)
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
df['Age'] = df['Age'].astype('int16')
df['Salary'] = df['Salary'].astype('float32')
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
            '''
        },
        "🔍 Advanced Topics": {
            "Time Series Analysis": '''
# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as index
df.set_index('Date', inplace=True)

# Resample data to monthly frequency and sum Sales
monthly_sales = df['Sales'].resample('M').sum()

# Plot resampled data
monthly_sales.plot(kind='line', title='Monthly Sales', figsize=(12,6))
plt.show()

# Rolling statistics: 12-month moving average
df['Sales_MA12'] = df['Sales'].rolling(window=12).mean()
df['Sales_MA12'].plot(title='12-Month Moving Average of Sales', figsize=(12,6))
plt.show()
            ''',
            "Handling Categorical Data": '''
# One-Hot Encoding using get_dummies
df_encoded = pd.get_dummies(df, columns=['Category'], drop_first=True)

# Label Encoding using sklearn
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Category_Encoded'] = le.fit_transform(df['Category'])

# Handling ordinal categories
ordinal_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
df['Priority_Level'] = df['Priority'].map(ordinal_mapping)
            ''',
            "Applying Functions with GroupBy": '''
# Define a custom aggregation function
def range_func(x):
    return x.max() - x.min()

# Apply custom function to 'Experience' column
experience_range = grouped_city['Experience'].apply(range_func)
print(experience_range)

# Apply multiple aggregation functions
aggregated = df.groupby('City').agg({
    'Salary': ['mean', 'sum', 'median'],
    'Age': 'max',
    'Experience': range_func
})
print(aggregated)
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
                st.markdown("<br>", unsafe_allow_html=True)
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
        <div style="background-color: #F5F5F5; color: black; text-align: center; padding: 20px; margin-top: 50px; border-top: 2px solid #CCCCCC;">
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
