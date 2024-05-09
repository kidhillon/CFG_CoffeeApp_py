# Authors: Kiran Dhillon and Cagla Tarioglu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
sns.set_theme(style="darkgrid")


# Step 1: Read the data from the spreadsheet using pandas (REQUIRED TASK)
def read_data():
    df = pd.read_csv('data/coffee.csv', encoding='latin-1')
    return df


# Step 2: Count number of coffees reviewed
def num_coffees(data):
    total_coffees = len(data)
    print("\nTotal number of coffees reviewed:", total_coffees)


# Step 3: Count number of unique countries of origin
def num_unique_countries(data):
    count_unique_countries = data['origin'].nunique()
    unique_countries = data['origin'].unique()
    print("Number of countries of origin:", count_unique_countries)
    print("\nAll Countries:", unique_countries)


# Step 4: Count how many coffees from each country (REQUIRED TASK)
def count_country_occurrence(data):
    country_counted = Counter(data['origin'])
    return country_counted


# Step 5: Plot graph showing each country and their number of coffees in data set
def plot_countries_num_coffees(country_counted):
    sorted_country_counts = sorted(country_counted.items(), key=lambda x: x[1], reverse=True)

    countries = [country_name for country_name, count in sorted_country_counts]
    counts = [count for country_total, count in sorted_country_counts]

    plt.figure(figsize=(10, 8))
    bars = plt.barh(countries, counts, color='brown')
    plt.xlabel('\nNumber of Coffees')
    plt.title('Number of Coffees from each country')
    plt.gca().invert_yaxis()  # Inverts y-axis to display countries from top to bottom
    for bar, count in zip(bars, counts):
        plt.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, str(count), ha='center', va='center')
    plt.subplots_adjust(left=0.2, right=0.9)
    plt.tight_layout()
    plt.show(block=False)


# Step 6: Calculate the average rating of coffee for each country (REQUIRED TASK)
def average_rating_by_country(data):
    country_average_rating = data.groupby('origin')['rating'].mean().to_dict()
    return country_average_rating


# Step 7: Calculate correlation between Prices of all coffees and rating and IF statements for relationship
def price_rating_correlation(data):
    price = data['100g_USD']
    rating = data['rating']

    correlation_coefficient = np.corrcoef(price, rating)[0, 1]
    print("\nCorrelation between Price per 100g and rating of coffee given:", correlation_coefficient)

    if correlation_coefficient > 0.8:
        print("There is a very strong positive correlation.")
    elif correlation_coefficient > 0.6:
        print("There is a strong positive correlation.")
    elif correlation_coefficient > 0.4:
        print("There is a moderate positive correlation.")
    elif correlation_coefficient > 0.2:
        print("There is a weak positive correlation.")
    elif -0.2 <= correlation_coefficient <= 0.2:
        print("There is no linear correlation.")
    elif -0.2 > correlation_coefficient >= -0.4:
        print("There is a weak negative correlation.")
    elif -0.4 > correlation_coefficient >= -0.6:
        print("There is a moderate negative correlation.")
    elif -0.6 > correlation_coefficient >= -0.8:
        print("There is a strong negative correlation.")
    elif correlation_coefficient < -0.8:
        print("There is a very strong negative correlation.")

# Step 8: Plot graph of price vs ratings
    log_price = np.log(price)  # logged price for better distribution on graph
    correlation_coefficient_square = correlation_coefficient ** 2

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=log_price,
        y=rating,
        label=f'Correlation coefficient (r) = {correlation_coefficient:.2f}, r^2 = {correlation_coefficient_square:.2f}'
    )
    sns.regplot(x=log_price, y=rating)

    plt.title('Prices of Coffee vs their Rating')
    plt.xlabel('Log(Price (USD) of Coffee per 100g)')
    plt.ylabel('Rating of coffees')
    plt.legend()
    plt.yticks(np.arange(min(rating), max(rating) + 1, 1))
    plt.show(block=False)


# Step 9: Ask user which roast type and generate word map with reviews from that roast type
# noinspection PyTypeChecker
def generate_roast_word_cloud():
    df = pd.read_csv("data/coffee.csv")

    roast_type = input(
        "\nEnter the roast type you want to create a word map from options "
        "(Light, Medium-Light, Medium, Medium-Dark or Dark): "
    )

    valid_roast_types = ['Light', 'Medium-Light', 'Medium', 'Medium-Dark', 'Dark']
    if roast_type not in valid_roast_types:
        print("\nInvalid roast type. Please choose from: Light, Medium-Light, Medium, Medium-Dark, Dark.")
        return

    review_data = f"review"
    review_data = ' '.join(df[review_data].astype(str))

    stopwords = set(STOPWORDS)
    stopwords.update(["aroma", "note", "cup", "flavor", "flavors", "hint", "toned", "structure",
                      "notes", "cacao", "cacao nib", "cocoa nib", "nib", "cocoa"]
                     )

    mask = np.array(Image.open('data/coffeemask.png'))

    wordcloud = WordCloud(
        stopwords=stopwords,
        width=2000,
        height=2000,
        colormap='Dark2',
        background_color='white',
        mask=mask,
        max_font_size=300,
    ).generate(review_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axis
    plt.title(f"Words describing '{roast_type}' roast coffees in reviews")
    plt.show()


# Execute functions in order
if __name__ == '__main__':
    # Read the data from CSV file
    coffee_data = read_data()
    print(coffee_data.head())

    # Calculate the number of coffees
    num_coffees(coffee_data)

    # Calculate the number of unique countries and list them
    num_unique_countries(coffee_data)

    # Plot bar graph of country occurrences
    country_counts = count_country_occurrence(coffee_data)
    plot_countries_num_coffees(country_counts)

    # Calculate average rating by country
    avg_rating_by_country_dict = average_rating_by_country(coffee_data)
    for country, avg_rating in avg_rating_by_country_dict.items():
        print(f"Average rating for {country}: {avg_rating:.2f}")

    # Calculate correlation between price and rating
    price_rating_correlation(coffee_data)

    # Generate wordcloud
    csv_file_path = "data/coffee.csv"
    generate_roast_word_cloud()
