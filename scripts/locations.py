from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import random
import re

# List of locations
locations = [
    # Transport
    "Airport Terminal 1 Indira Gandhi International",
    "Airport Terminal 3 Indira Gandhi International",
    "Airport Terminal 2 Indira Gandhi International",
    "Hazrat Nizamuddin",
    "Sarai Kale Khan ISBT",
    "Kashmiri Gate ISBT (Maharana Pratap)",
    "New Delhi Railway Station",
    "Old Delhi Railway Station",
    "Delhi Sarai Rohilla",
    "Delhi Cantt Railway Station",
    "Anand Vihar Railway Station",
    "Kaushambi bus stand (ISBT Anand Vihar)",
    "Dhaula Kuan Bus Stand",
    # Shopping, Food & Nightlife
    "Connaught Place",
    # Iconic Structures
    "Red Fort",
    "Rajghat Mahatma Gandhi memorial",
    "Agrasen ki Baoli",
    "Jantar Mantar",
    "Rashtrapati Bhavan",
    "India Gate",
    "Safdarjung Tomb, Delhi",
    "Humayun’s Tomb",
    "Qutub Minar",
    "Feroz Shah Kotla Fort",
    "Hauz Khas Village complex Deer park",
    "Purana Quila",
    "Tughlaqabad Fort",
    "Mehrauli Archaeological Park",
    # Spiritual Sites
    "Jama Masjid",
    "Akshardham temple",
    "Dargah Nizamuddin Aulia",
    "Lotus Temple",
    "ISKCON Temple",
    "Hanuman Mandir Karol Bagh",
    "Gurudwara Sri Bangla Sahib",
    "Shri Laxmi Narayan Temple Birla Mandir",
    "Chattarpur Mandir",
    "Tibetan Monastery Majnu ka Tila",
    # Art and Museum
    "National Science Centre, Delhi",
    "National Crafts Museum & Hastkala Academy",
    "Shankar's International Dolls Museum",
    "National Museum, New Delhi",
    "National Rail Museum",
    "Sulabh International Museum Of Toilets",
    "Sanskriti Kendra",
    "Indira Gandhi Memorial Museum, Delhi",
    "Nehru Planetarium",
    "Prime Minister Museum & Library",
    "Museum of Illusions",
    # Nature and Park
    "National Zoological Park",
    "Okhla Bird Sanctuary",
    "Central Ridge Forest",
    "Lodhi Garden",
    "Sanjay Van",
    "The Garden of Five Senses",
    "Indraprastha Park",
    "Nehru Park"
]

def is_valid_maps_url(url):
    """Check if the URL is a valid Google Maps URL."""
    return bool(re.match(r"https://www\.google\.com/maps/place/.*", url))

def scrape_and_visit_google_maps_urls(locations, max_retries=3, headless=False):
    # Path to ChromeDriver (update as needed)
    PATH = r"C:\Program Files (x86)\chromedriver.exe"
    service = Service(PATH)

    # Setup Chrome options
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    # Initialize WebDriver
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.execute_cdp_cmd("Network.setUserAgentOverride", {
        "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    results = []

    for location in locations:
        print(f"Searching for {location} on Google Maps...")
        attempt = 0
        place_url = None

        # Scrape URL
        while attempt < max_retries and not place_url:
            try:
                # Open Google Maps
                driver.get("https://www.google.com/maps")
                time.sleep(random.uniform(3, 7))

                # Wait for search bar
                wait = WebDriverWait(driver, 15)
                search_bar = wait.until(EC.presence_of_element_located((By.ID, "searchboxinput")))

                # Input location with random typing delay
                search_bar.clear()
                for char in location:
                    search_bar.send_keys(char)
                    time.sleep(random.uniform(0.05, 0.2))
                search_bar.send_keys(Keys.RETURN)

                # Wait for results
                time.sleep(random.uniform(5, 10))

                # Get current URL and validate
                current_url = driver.current_url
                if is_valid_maps_url(current_url):
                    place_url = current_url
                    print(f"Found URL for {location}: {place_url}")
                else:
                    print(f"Invalid URL for {location} on attempt {attempt + 1}: {current_url}")
                    attempt += 1
                    time.sleep(random.uniform(5, 10))  # Wait before retrying

            except Exception as e:
                print(f"Error for {location} on attempt {attempt + 1}: {e}")
                attempt += 1
                time.sleep(random.uniform(5, 10))  # Wait before retrying

        # Initialize result dictionary
        result = {
            "place_name": location,
            "place_url": place_url if place_url else "Not found",
            "visit_status": "Not visited",
            "page_title": "N/A"
        }

        # Visit the scraped URL if valid
        if place_url and is_valid_maps_url(place_url):
            print(f"Visiting URL for {location}: {place_url}")
            try:
                driver.get(place_url)
                time.sleep(random.uniform(3, 7))  # Wait for page to load

                # Wait for the page to be fully loaded
                wait = WebDriverWait(driver, 15)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "title")))

                # Get page title to confirm the page loaded correctly
                page_title = driver.title
                result["visit_status"] = "Success"
                result["page_title"] = page_title if page_title else "No title"
                print(f"Successfully visited {location}. Page title: {page_title}")

            except Exception as e:
                result["visit_status"] = f"Failed: {str(e)}"
                print(f"Failed to visit URL for {location}: {e}")

            time.sleep(random.uniform(2, 5))  # Delay after visiting

        results.append(result)
        time.sleep(random.uniform(2, 5))  # Delay between searches

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv("scraped_and_visited_google_maps_urls.csv", index=False)
        print("✅ Results saved to scraped_and_visited_google_maps_urls.csv!")
    else:
        print("❌ No results were successfully scraped or visited.")

    driver.quit()

if __name__ == "__main__":
    scrape_and_visit_google_maps_urls(locations, headless=False)  # Set headless=True for headless mode