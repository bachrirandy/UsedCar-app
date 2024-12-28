import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

st.title('Submission Machine Learning GDGoC')
st.write("Mochammad Randy Surya Bachri")

st.header("About Dataset")
st.write("This dataset is a collection of data from Kaggle, which contains information about used cars in the Indian market, featuring 9,582 records with 11 detailed attributes. Collected up to November 2024, it offers an in-depth overview of India's second-hand car market.")
st.write("Dataset Links: [Data Science Salaries Dataset](https://www.kaggle.com/datasets/yusufdelikkaya/datascience-salaries-2024)")

st.header("1. Data Wrangling")
st.subheader("Displaying the Top 5 and Bottom 5 Rows of the Dataset")

df = pd.read_csv('https://drive.google.com/uc?id=1C9O5pNPeg1PK_qlLxa2Ow2FYCUelv0mr')
data = pd.concat([df.head(5), df.tail(5)])
st.write(data)

st.subheader("Column Description")
st.write("""
         1. Brand: Car manufacturer (e.g., Volkswagen, Maruti Suzuki, Honda, Tata).
         2. Model: Specific car model (e.g., Taigun, Baleno, Polo, WRV).
         3. Year: Manufacturing year of the vehicle (ranging from older models to 2024).
         4. Age: Age of the vehicle in years.
         5. kmDriven: Total kilometers driven by the vehicle.
         6. Transmission: Type of transmission (Manual or Automatic).
         7. Owner: Ownership status (first or second owner).
         8. FuelType: Type of fuel (Petrol, Diesel, Hybrid/CNG).
         9. PostedDate: When the car listing was posted.
         10.AdditionalInfo: Extra details about the vehicle.
         """)

st.subheader("Dataset Information")
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

st.write("""The dataset contains **9,582 entries** with **11 columns**
         The dataset consists of:
- **2 integer columns**
- **9 object (string) columns**

The total memory usage is approximately **823.6 KB**.
""")

st.header("2. Data Availibility")

st.subheader("Missing Values")
st.write(df.isnull().sum())
st.write("""
The dataset contains missing values in the following column:
- **kmDriven**: 47 missing entries

All other columns have complete data (no missing values).
""")

st.subheader("Data Cleaning and Convertion")
# Membersihkan dan mengonversi kolom kmDriven
df['kmDriven'] = df['kmDriven'].str.replace(' km', '').str.replace(',', '').astype('float')

# Membersihkan dan mengonversi kolom AskPrice
df['AskPrice'] = df['AskPrice'].str.replace('₹ ', '').str.replace(',', '').astype('int64')

# Mengonversi kolom PostedDate ke format datetime
df['PostedDate'] = pd.to_datetime(df['PostedDate'], format='%b-%y', errors='coerce')

# Menangani nilai yang hilang pada kolom kmDriven (mengisi dengan median)
median_km = df['kmDriven'].median()
df['kmDriven'].fillna(median_km, inplace=True)

# Memeriksa nilai yang hilang setelah pembersihan
null_values_after_cleaning = df.isnull().sum()

# Menampilkan informasi DataFrame dan status nilai null
st.write("DataFrame Information after cleaning:")
st.write(df.info())

st.write("null after cleaning:")
st.write(null_values_after_cleaning)

st.write("Cleaning and preparing the data for analysis. Certain columns containing text or currency symbols are converted to numeric data types, date columns are converted to datetime types, and missing values ​​in the kmDriven column are filled with the median. After cleaning, information about the DataFrame and the status of missing values ​​is displayed to verify the results.")

st.subheader("Duplicate Rows Check")
duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    st.write(f"The dataset contains **{duplicate_count} duplicate rows**.")
    # Menampilkan baris duplikat jika ada
    st.write("Here are the duplicate rows:")
    st.write(df[df.duplicated()])
else:
    st.write("The dataset contains no duplicate rows.")
    
st.header("3. Exploratory Data Analysis")
st.subheader("Descriptive Statistic")

st.write(df.describe(include='all'))
st.write("""
The dataset contains **8,851 records** with **11 columns**.

### <> Numerical Columns:
- **Year**: The cars range from **1986 to 2024**, with a mean year of around **2016**.
- **Age**: Cars range in age from **0 to 38 years**, with an average age of approximately **7.6 years**.
- **kmDriven**: The mileage varies widely, with the range from **0 to 980,002 km**. The mean mileage is **around 70,979 km**, with cars typically having between **43,327 km and 86,000 km** driven.

### <> Categorical Columns:
- **Brand**: There are **39 unique brands** in the dataset, with **Maruti Suzuki** being the most frequent, appearing **2,589 times**.
- **Model**: There are **400 unique models**, with **City** being the most common model, appearing **306 times**.
- **Transmission**: The most common transmission type is **Manual**, with **4,662 occurrences**.
- **Owner**: The majority of cars are listed as **first-owner**, with **4,592 occurrences**.
- **FuelType**: The most frequent fuel type is **Diesel**, with **3,517 occurrences**.

""")


st.subheader("Correlation Heatmap of Year, Age, kmDriven, and AskPrice")

# Menghitung matriks korelasi
correlation_matrix = df[['Year', 'Age', 'kmDriven', 'AskPrice']].corr()

# Membuat heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Year, Age, kmDriven, and AskPrice")

# Menampilkan heatmap di Streamlit
st.pyplot(plt)

st.write("""
         1. The correlation value between Year and Age of -1.00 indicates a very strong and negative relationship between the year of manufacture and the age of the car.
         2. The correlation value between Year and AskPrice of 0.30 shows a positive relationship, indicating that newer cars tend to have higher prices.
         3. The correlation value between Age and AskPrice of -0.30 indicates a negative relationship, suggesting that older cars tend to have lower prices.
         4. kmDriven has a weak correlation with Year (-0.28), Age (0.28), and AskPrice (-0.14), indicating that the number of kilometers driven by the car does not significantly affect its price or other features.
         """)


st.subheader("Average Ask Price by Transmission")

# Membuat plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Transmission', y='AskPrice', data=df, palette="inferno")

# Menambahkan garis rata-rata
average_price = df['AskPrice'].mean()
plt.axhline(y=average_price, color='red', linestyle='--', label=f'Average: {average_price:.2f}')

# Menambahkan detail ke plot
plt.title("Average Ask Price by Transmission")
plt.xlabel("Transmission")
plt.ylabel("Ask Price")
plt.legend()

# Menampilkan plot di Streamlit
st.pyplot(plt)

st.write("Cars with Automatic transmissions have a much higher average price than cars with Manual transmissions.")

top_10_brands = df['Brand'].value_counts().head(10).index

# Membuat visualisasi
st.write("### Top 10 Sold by Car Brand with Transmission")

plt.figure(figsize=(20, 10))
sns.countplot(x='Brand', hue='Transmission', data=df, palette="inferno", order=top_10_brands)
plt.title("Top 10 Sold by Car Brand with Transmission")
plt.xlabel("Brand")
plt.ylabel("Count")

# Menampilkan plot di Streamlit
st.pyplot(plt)

st.write("""
         - Manual transmission cars dominate sales for some brands, especially Maruti Suzuki, which has the highest sales figures compared to other brands.
         - Premium brands such as BMW, Mercedes-Benz, and Audi show the dominance of Automatic transmission.
         - Brands like Maruti Suzuki, Hyundai, and Honda have higher sales volumes overall, but most of their sales come from manual transmissions.
         In contrast, premium brands like BMW and Mercedes-Benz have smaller sales volumes, but almost entirely in the automatic transmission category.
         """)


# Membuat visualisasi rata-rata harga berdasarkan tahun
st.subheader("Average Ask Price by Year")

plt.figure(figsize=(20, 6))
sns.barplot(x='Year', y='AskPrice', data=df, palette="inferno")

# Menambahkan garis rata-rata
average_price = df['AskPrice'].mean()
plt.axhline(y=average_price, color='red', linestyle='--', label=f'Average Price: {average_price:.2f}')

# Menambahkan detail ke plot
plt.title("Average Ask Price by Year")
plt.xlabel("Year")
plt.ylabel("Ask Price")
plt.legend()

average_year = df['Year'].mean()
st.write(f"Average Year of Cars: {average_year:.2f}")

# Menampilkan plot di Streamlit
st.pyplot(plt)

st.write("Cars from newer years (2020 and above) have higher prices than older cars. Cars from 1999 seem to be an anomaly with a very high average price, possibly due to the premium or scarcity of certain models.")


# Membuat visualisasi rata-rata harga berdasarkan umur mobil
st.subheader("Average Ask Price by Age")

plt.figure(figsize=(20, 6))
sns.barplot(x='Age', y='AskPrice', data=df, palette="inferno")

# Menambahkan garis rata-rata
average_price = df['AskPrice'].mean()
plt.axhline(y=average_price, color='red', linestyle='--', label=f'Average Price: {average_price:.2f}')

# Menambahkan detail ke plot
plt.title("Average Ask Price by Age")
plt.xlabel("Age")
plt.ylabel("Ask Price")
plt.legend()

# Menampilkan rata-rata umur mobil
average_age = df['Age'].mean()
st.write(f"Average Age of Cars: {average_age:.2f}")

# Menampilkan plot di Streamlit
st.pyplot(plt)

st.write("""
         - New cars (0-2 years old) have a higher average price. 
         - Cars with an age of 10 years and above are cheaper.
         - Cars around 25 years old have high prices. This is likely due to classic cars having historical value and high demand.
         """)

# Judul aplikasi
st.subheader("Average Ask Price by Fuel Type")

# Menghitung rata-rata harga keseluruhan
average_price = df['AskPrice'].mean()

# Membuat visualisasi dengan Matplotlib dan Seaborn
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='FuelType', y='AskPrice', data=df, palette="inferno", ax=ax)
ax.axhline(y=average_price, color='red', linestyle='--', label=f'Average: {average_price:.2f}')
ax.set_title("Average Ask Price by Fuel Type")
ax.set_xlabel("Fuel Type")
ax.set_ylabel("Ask Price")
ax.legend()

# Menampilkan plot di Streamlit
st.pyplot(fig)

st.write("""
         1. Diesel has the highest average price, around 1.4 million, which is higher than the overall average. 
         2. Petrol is in the middle position, with an average price slightly below the overall average.
         3. Hybrid/CNG has the lowest average price among the other fuel types, at under 1 million. 
         """)

# Judul aplikasi
st.subheader("Top 10 Sold by Car Brand with Fuel Type")

# Ambil 10 brand teratas berdasarkan jumlah penjualan
top_10_brands = df['Brand'].value_counts().head(10).index

# Membuat visualisasi dengan Matplotlib dan Seaborn
fig, ax = plt.subplots(figsize=(20, 10))
sns.countplot(x='Brand', hue='FuelType', data=df, palette="inferno", order=top_10_brands, ax=ax)
ax.set_title("Top 10 Sold by Car Brand with Fuel Type")
ax.set_xlabel("Brand")
ax.set_ylabel("Count")
ax.legend(title="Fuel Type")

# Menampilkan plot di Streamlit
st.pyplot(fig)

st.write("""
         1. Petrol: Maruti Suzuki dominates sales in the Petrol segment, followed by Hyundai. The fuel is popular in the mass-market and premium segments, including in Mercedes-Benz and BMW.
         2. Diesel: Toyota leads in the Diesel segment. Mahindra, Volkswagen and Audi also have significant Diesel sales.
         3. Hybrid/CNG: Maruti Suzuki leads in the Hybrid/CNG segment.
         """)

st.subheader("Brands by Average Ask Price and Number of Cars Sold")

avg_price_by_brand = df.groupby('Brand')['AskPrice'].mean().sort_values(ascending=False)
brand_counts = df['Brand'].value_counts().loc[avg_price_by_brand.index]

plt.figure(figsize=(12, 6))

ax1 = sns.barplot(x=avg_price_by_brand.index, y=avg_price_by_brand.values, palette="inferno")

ax2 = ax1.twinx()
sns.lineplot(x=brand_counts.index, y=brand_counts.values, marker='o', ax=ax2, color='red', label='Number of Cars Sold', linewidth=2)

ax1.set_title("Brands by Average Ask Price and Number of Cars Sold")
ax1.set_xlabel("Brand")
ax1.set_ylabel("Average Price (₹)")
ax2.set_ylabel("Number of Cars Sold")

ax1.tick_params(axis='x', rotation=90)

ax1.legend(labelspacing=1.2)
ax2.legend(labelspacing=1.2)

st.pyplot(plt)

st.write("Premium brands (e.g., Aston Martin, Rolls Royce) have high prices, but low sales. Popular and economical brands (e.g. Suzuki, Maruti, Toyota) have low prices with much higher sales volumes.")

st.subheader("Insight Conclusion")

st.write("""
         - The year of production and age of the car have a big influence on the price. Newer cars (0-2 years old) tend to have higher prices, while older cars (10+ years old) are cheaper, except for classic cars that have historical value.
         - Diesel cars have the highest average price, followed by gasoline, while hybrid/CNG have lower prices.
         - Cars with automatic transmissions have a higher average price compared to manual transmissions. However, manual transmissions are more widely used, especially by mass-market brands like Maruti Suzuki and Hyundai.
         - Maruti Suzuki leads the market with the highest sales in both the gasoline and hybrid/CNG segments. Toyota excels in the diesel segment.
         - Premium brands have much higher prices, but lower sales volumes compared to popular affordable brands like Maruti Suzuki and Hyundai.
         - Consumers tend to choose cars that are affordable and fuel-efficient. Cars with manual transmissions and gasoline fuel are more sold. 
         
         """)
