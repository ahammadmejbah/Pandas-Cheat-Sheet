
import streamlit as st
import base64
import requests
from streamlit_lottie import st_lottie
import json

# Initial page config
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
    lottie_header = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_ydo1amjm.json")
    lottie_section = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_1pxqjqps.json")
    
    # Header with animation
    st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #FF4B4B;">📊 Pandas Cheat Sheet By Mejbah Ahammad</h1>
        </div>
    """, unsafe_allow_html=True)
    
    if lottie_header:
        st_lottie(lottie_header, height=200, key="header")

    # Define Pandas topics and their code snippets
    sections = {
        "📦 Importing & Setup": {
            "Importing Libraries": '''
import pandas as pd
import numpy as np
    ''',
            "Reading Data": '''
# Read CSV
df = pd.read_csv('data.csv')

# Read Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Read JSON
df = pd.read_json('data.json')

# Read from SQL
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:password@localhost:5432/mydatabase')
df = pd.read_sql('SELECT * FROM table_name', engine)
    ''',
            "Basic Data Inspection": '''
# View first 5 rows
df.head()

# View last 5 rows
df.tail()

# Get DataFrame info
df.info()

# Summary statistics
df.describe()

# Check for missing values
df.isnull().sum()
    '''
        },
        "🔍 Data Exploration": {
            "Selecting Columns": '''
# Select single column
df['Age']

# Select multiple columns
df[['Name', 'Age', 'Salary']]
    ''',
            "Filtering Rows": '''
# Filter rows based on condition
df[df['Age'] > 30]

# Multiple conditions
df[(df['Age'] > 30) & (df['Salary'] > 50000)]
    ''',
            "Sorting Data": '''
# Sort by single column
df.sort_values(by='Age', ascending=False)

# Sort by multiple columns
df.sort_values(by=['City', 'Age'], ascending=[True, False])
    ''',
            "Handling Missing Values": '''
# Drop rows with any missing values
df.dropna(inplace=True)

# Fill missing values with a specific value
df.fillna(0, inplace=True)

# Fill missing values with mean of the column
df['Age'].fillna(df['Age'].mean(), inplace=True)
    '''
        },
        "🔄 Data Transformation": {
            "Applying Functions": '''
# Apply function to a column
df['Age'] = df['Age'].apply(lambda x: x + 1)

# Apply function to entire DataFrame
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    ''',
            "Vectorized Operations": '''
# Create new column based on existing columns
df['Salary_Per_Age'] = df['Salary'] / df['Age']

# Vectorized string operations
df['Name'] = df['Name'].str.upper()
    ''',
            "Mapping Values": '''
# Map values using a dictionary
df['City'] = df['City'].map({'New York': 'NY', 'Los Angeles': 'LA', 'Chicago': 'CHI'})
    ''',
            "Binning Data": '''
# Binning numerical data
df['Age Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])
    '''
        },
        "🔗 Merging & Joining": {
            "Merging DataFrames": '''
# Inner join
merged_df = pd.merge(df1, df2, on='Key', how='inner')

# Left join
merged_df = pd.merge(df1, df2, on='Key', how='left')

# Right join
merged_df = pd.merge(df1, df2, on='Key', how='right')

# Outer join
merged_df = pd.merge(df1, df2, on='Key', how='outer')
    ''',
            "Concatenating DataFrames": '''
# Concatenate vertically
concatenated_df = pd.concat([df1, df2, df3], axis=0)

# Concatenate horizontally
concatenated_df = pd.concat([df1, df2, df3], axis=1)
    ''',
            "Joining on Multiple Keys": '''
# Merge on multiple keys
merged_df = pd.merge(df1, df2, on=['Key1', 'Key2'], how='inner')
    '''
        },
        "📊 Grouping & Aggregation": {
            "Group By": '''
# Group by single column
grouped = df.groupby('City')

# Group by multiple columns
grouped = df.groupby(['City', 'Age Group'])
    ''',
            "Aggregation Functions": '''
# Aggregate with mean
grouped['Salary'].mean()

# Multiple aggregations
grouped.agg({'Age': ['mean', 'sum'], 'Salary': 'median'})

# Custom aggregation
grouped.agg({'Salary': ['mean', 'sum'], 'Experience': lambda x: x.max() - x.min()})
    ''',
            "Pivot Tables": '''
# Simple pivot table
pivot = df.pivot_table(values='Sales', index='Region', columns='Product', aggfunc='sum', fill_value=0)

# Pivot with multiple aggregation functions
pivot = df.pivot_table(values='Sales', index='Region', columns='Product', aggfunc=['sum', 'mean'], fill_value=0)

# Pivot with margins (totals)
pivot = df.pivot_table(values='Sales', index='Region', columns='Product', aggfunc='sum', margins=True, fill_value=0)
    '''
        },
        "📈 Data Visualization": {
            "Plotting with Pandas": '''
# Line plot
df.plot(kind='line', x='Date', y='Sales', title='Sales Over Time')

# Bar plot
df.plot(kind='bar', x='City', y='Sales', title='Sales by City')

# Scatter plot
df.plot(kind='scatter', x='Age', y='Salary', title='Age vs Salary')

# Histogram
df['Age'].plot(kind='hist', bins=10, title='Age Distribution')

# Boxplot
df.boxplot(column='Salary', by='City', title='Salary Distribution by City')
    ''',
            "Advanced Visualization": '''
# Using seaborn for enhanced visuals
import seaborn as sns

# Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# Pairplot
sns.pairplot(df, hue='City')

# Boxplot
sns.boxplot(x='City', y='Salary', data=df)

# Violin plot
sns.violinplot(x='City', y='Salary', data=df)
    '''
        },
        "🔧 Data Engineering": {
            "Exporting Data": '''
# Export to CSV
df.to_csv('output.csv', index=False)

# Export to Excel
df.to_excel('output.xlsx', index=False, sheet_name='Sheet1')

# Export to JSON
df.to_json('output.json', orient='records', lines=True)
    ''',
            "Handling Large Datasets": '''
# Read large CSV in chunks
chunksize = 10**6
for chunk in pd.read_csv('large_data.csv', chunksize=chunksize):
    process(chunk)

# Optimize memory usage
df['Age'] = df['Age'].astype('int8')
df['Salary'] = df['Salary'].astype('float32')
    ''',
            "Using SQL with Pandas": '''
# Querying SQL database
query = "SELECT * FROM table_name WHERE Age > 30"
df_filtered = pd.read_sql(query, engine)
    '''
        },
        "🔍 Advanced Topics": {
            "Time Series Analysis": '''
# Convert to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set index
df.set_index('Date', inplace=True)

# Resample data
monthly_sales = df['Sales'].resample('M').sum()

# Rolling statistics
df['Sales_MA'] = df['Sales'].rolling(window=12).mean()
    ''',
            "Handling Categorical Data": '''
# One-Hot Encoding
df = pd.get_dummies(df, columns=['Category'])

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])
    ''',
            "Applying Functions with GroupBy": '''
# Apply custom function
def custom_func(x):
    return x.max() - x.min()

result = grouped['Experience'].apply(custom_func)
    '''
        }
    }

    # Load section animation if available
    if lottie_section:
        st_lottie(lottie_section, height=150, key="section")

    # Render sections using tabs
    st.header("📚 Pandas Topics")
    tabs = st.tabs(list(sections.keys()))
    for tab, (section_title, subtopics) in zip(tabs, sections.items()):
        with tab:
            # Display section animation
            if lottie_section:
                st_lottie(lottie_section, height=100, key=f"{section_title}_animation")
            
            # Display subtopics using expanders
            for sub_title, code in subtopics.items():
                with st.expander(sub_title, expanded=False):
                    # Display code with appropriate syntax highlighting
                    language = 'python'  # Pandas uses Python
                    st.code(code, language=language)

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
            <small>Pandas Cheat Sheet v1.0 | Nov 2024 | <a href="https://ahammadmejbah.com/" style="color: #000000;">Mejbah Ahammad</a></small>
            <div class="card-footer">Mejbah Ahammad © 2024</div>
        </div>
    """, unsafe_allow_html=True)

    # Optional: Add some spacing at the bottom
    st.markdown("<br><br><br>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
