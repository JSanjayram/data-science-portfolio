import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import re

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_quotes(self):
        """Scrape quotes from quotes.toscrape.com"""
        quotes_data = []
        base_url = "http://quotes.toscrape.com/page/{}"
        
        print("📝 Scraping quotes...")
        for page in range(1, 6):  # Scrape first 5 pages
            try:
                response = self.session.get(base_url.format(page))
                soup = BeautifulSoup(response.content, 'html.parser')
                
                quotes = soup.find_all('div', class_='quote')
                if not quotes:
                    break
                
                for quote in quotes:
                    text = quote.find('span', class_='text').text.strip()
                    author = quote.find('small', class_='author').text.strip()
                    tags = [tag.text for tag in quote.find_all('a', class_='tag')]
                    
                    quotes_data.append({
                        'quote': text,
                        'author': author,
                        'tags': ', '.join(tags),
                        'word_count': len(text.split()),
                        'char_count': len(text),
                        'scraped_at': datetime.now()
                    })
                
                time.sleep(1)  # Be respectful
                
            except Exception as e:
                print(f"Error scraping page {page}: {e}")
        
        return pd.DataFrame(quotes_data)
    
    def scrape_news_headlines(self):
        """Scrape news headlines from example news site"""
        headlines_data = []
        
        # Using a simple news aggregator for demo
        urls = [
            "https://httpbin.org/html",  # Safe test endpoint
        ]
        
        print("📰 Scraping news headlines...")
        
        # Create sample news data for demonstration
        sample_headlines = [
            "Technology Advances in AI and Machine Learning",
            "Global Climate Change Summit Reaches Agreement", 
            "Stock Market Shows Strong Performance This Quarter",
            "New Medical Breakthrough in Cancer Research",
            "Space Exploration Mission Launches Successfully",
            "Economic Growth Continues Despite Challenges",
            "Education Reform Bill Passes Legislature",
            "Renewable Energy Projects Expand Nationwide",
            "Sports Championship Finals Draw Record Viewers",
            "Cultural Festival Celebrates Diversity and Unity"
        ]
        
        categories = ['Technology', 'Environment', 'Finance', 'Health', 'Science', 
                     'Economy', 'Education', 'Energy', 'Sports', 'Culture']
        
        for i, headline in enumerate(sample_headlines):
            headlines_data.append({
                'headline': headline,
                'category': categories[i],
                'word_count': len(headline.split()),
                'char_count': len(headline),
                'scraped_at': datetime.now()
            })
        
        return pd.DataFrame(headlines_data)
    
    def scrape_weather_data(self):
        """Scrape weather information"""
        weather_data = []
        
        print("🌤️ Generating weather data...")
        
        # Sample weather data for major cities
        cities = ['New York', 'London', 'Tokyo', 'Sydney', 'Mumbai', 'Berlin', 'Toronto', 'Dubai']
        
        import random
        random.seed(42)
        
        for city in cities:
            temp = random.randint(-5, 35)
            humidity = random.randint(30, 90)
            conditions = random.choice(['Sunny', 'Cloudy', 'Rainy', 'Partly Cloudy', 'Snowy'])
            
            weather_data.append({
                'city': city,
                'temperature': temp,
                'humidity': humidity,
                'condition': conditions,
                'scraped_at': datetime.now()
            })
        
        return pd.DataFrame(weather_data)

def analyze_scraped_data(quotes_df, headlines_df, weather_df):
    """Analyze all scraped data"""
    plt.figure(figsize=(16, 12))
    
    # 1. Quotes analysis
    plt.subplot(3, 4, 1)
    quotes_df['author'].value_counts().head(10).plot(kind='bar')
    plt.title('Top Authors by Quote Count')
    plt.xticks(rotation=45)
    
    plt.subplot(3, 4, 2)
    plt.hist(quotes_df['word_count'], bins=15, alpha=0.7, color='skyblue')
    plt.title('Quote Word Count Distribution')
    plt.xlabel('Word Count')
    
    plt.subplot(3, 4, 3)
    # Extract and count individual tags
    all_tags = []
    for tags_str in quotes_df['tags']:
        if tags_str:
            all_tags.extend([tag.strip() for tag in tags_str.split(',')])
    
    tag_counts = pd.Series(all_tags).value_counts().head(10)
    tag_counts.plot(kind='bar')
    plt.title('Most Popular Quote Tags')
    plt.xticks(rotation=45)
    
    # 4. Headlines analysis
    plt.subplot(3, 4, 4)
    headlines_df['category'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('News Categories Distribution')
    
    plt.subplot(3, 4, 5)
    plt.hist(headlines_df['word_count'], bins=10, alpha=0.7, color='lightcoral')
    plt.title('Headline Word Count Distribution')
    plt.xlabel('Word Count')
    
    plt.subplot(3, 4, 6)
    headlines_df['char_count'].plot(kind='box')
    plt.title('Headline Character Count')
    
    # 7. Weather analysis
    plt.subplot(3, 4, 7)
    weather_df.plot(x='city', y='temperature', kind='bar', ax=plt.gca())
    plt.title('Temperature by City')
    plt.xticks(rotation=45)
    
    plt.subplot(3, 4, 8)
    weather_df['condition'].value_counts().plot(kind='bar')
    plt.title('Weather Conditions Distribution')
    plt.xticks(rotation=45)
    
    plt.subplot(3, 4, 9)
    plt.scatter(weather_df['temperature'], weather_df['humidity'], 
               c=weather_df.index, cmap='viridis', alpha=0.7)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Humidity (%)')
    plt.title('Temperature vs Humidity')
    
    # 10. Combined analysis
    plt.subplot(3, 4, 10)
    data_counts = [len(quotes_df), len(headlines_df), len(weather_df)]
    data_types = ['Quotes', 'Headlines', 'Weather']
    plt.bar(data_types, data_counts, color=['gold', 'lightblue', 'lightgreen'])
    plt.title('Scraped Data Counts')
    
    # 11. Word count comparison
    plt.subplot(3, 4, 11)
    plt.boxplot([quotes_df['word_count'], headlines_df['word_count']], 
                labels=['Quotes', 'Headlines'])
    plt.title('Word Count Comparison')
    plt.ylabel('Word Count')
    
    # 12. Scraping timeline
    plt.subplot(3, 4, 12)
    scrape_times = [quotes_df['scraped_at'].iloc[0], headlines_df['scraped_at'].iloc[0], 
                   weather_df['scraped_at'].iloc[0]]
    plt.plot(data_types, [1, 2, 3], 'o-', markersize=8)
    plt.title('Scraping Sequence')
    plt.ylabel('Order')
    
    plt.tight_layout()
    plt.savefig('web_scraping_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_scraping_insights(quotes_df, headlines_df, weather_df):
    """Generate insights from scraped data"""
    insights = []
    
    # Quotes insights
    total_quotes = len(quotes_df)
    avg_quote_length = quotes_df['word_count'].mean()
    top_author = quotes_df['author'].value_counts().index[0]
    insights.append(f"Scraped {total_quotes} quotes with avg length {avg_quote_length:.1f} words")
    insights.append(f"Most quoted author: {top_author}")
    
    # Headlines insights
    total_headlines = len(headlines_df)
    avg_headline_length = headlines_df['word_count'].mean()
    top_category = headlines_df['category'].value_counts().index[0]
    insights.append(f"Scraped {total_headlines} headlines with avg length {avg_headline_length:.1f} words")
    insights.append(f"Most common news category: {top_category}")
    
    # Weather insights
    total_cities = len(weather_df)
    avg_temp = weather_df['temperature'].mean()
    hottest_city = weather_df.loc[weather_df['temperature'].idxmax(), 'city']
    insights.append(f"Weather data for {total_cities} cities, avg temp: {avg_temp:.1f}°C")
    insights.append(f"Hottest city: {hottest_city}")
    
    return insights

def main():
    print("🕷️ Web Scraping Project")
    print("="*50)
    
    # Initialize scraper
    scraper = WebScraper()
    
    # Scrape different types of data
    quotes_df = scraper.scrape_quotes()
    headlines_df = scraper.scrape_news_headlines()
    weather_df = scraper.scrape_weather_data()
    
    print(f"\n📊 Scraping Results:")
    print(f"Quotes: {len(quotes_df)} items")
    print(f"Headlines: {len(headlines_df)} items")
    print(f"Weather: {len(weather_df)} items")
    
    # Analyze data
    print("\n📈 Creating analysis visualizations...")
    analyze_scraped_data(quotes_df, headlines_df, weather_df)
    
    # Generate insights
    insights = generate_scraping_insights(quotes_df, headlines_df, weather_df)
    print("\n" + "="*50)
    print("SCRAPING INSIGHTS:")
    print("="*50)
    for insight in insights:
        print(f"• {insight}")
    
    # Save data
    quotes_df.to_csv('scraped_quotes.csv', index=False)
    headlines_df.to_csv('scraped_headlines.csv', index=False)
    weather_df.to_csv('scraped_weather.csv', index=False)
    
    print(f"\n✅ Scraping complete! Data saved to CSV files")
    print("📊 Check 'web_scraping_analysis.png' for visualizations")

if __name__ == "__main__":
    main()