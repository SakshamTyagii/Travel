from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
import pandas as pd
import time

# Path to your ChromeDriver
PATH = r"C:\Program Files (x86)\chromedriver.exe"

# Initialize the WebDriver with the specified ChromeDriver path
service = Service(PATH)
driver = webdriver.Chrome(service=service)

# Read the CSV file from the scripts folder
csv_file = "loc.csv"  # Assumes the script is in the scripts folder
df = pd.read_csv(csv_file)

# List to store results
results = []

# Open Google Maps
driver.get("https://www.google.com/maps")
# Wait for the search bar to be present
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "searchboxinput")))

# Iterate through each location in the CSV
for index, row in df.iterrows():
    location = row["Location"]
    city = row["City"]
    search_query = f"{location}, {city}, India"
    
    try:
        # Find the search bar and enter the query
        search_bar = driver.find_element(By.ID, "searchboxinput")
        search_bar.clear()
        search_bar.send_keys(search_query)
        search_bar.send_keys(Keys.ENTER)
        
        # Wait for the place page to load (look for the tabs section)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "RWPxGd")))
        
        # Find and click the "Reviews" tab
        review_buttons = driver.find_elements(By.XPATH, "//button[contains(@aria-label, 'Reviews') and contains(@class, 'hh2c6')]")
        review_found = False
        for button in review_buttons:
            button_aria_label = button.get_attribute("aria-label")
            if button_aria_label and "Reviews" in button_aria_label:
                button.click()
                time.sleep(3)  # Wait for the reviews page to load
                review_found = True
                break
        
        if not review_found:
            print(f"No Reviews tab found for {search_query}")
            # Navigate back to the search page
            driver.get("https://www.google.com/maps")
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "searchboxinput")))
            continue

        # Scrape the current URL (should be the reviews page)
        review_page_url = driver.current_url
        results.append({
            "City": city,
            "Location": location,
            "URL": review_page_url
        })
        print(f"Successfully scraped review URL for {search_query}: {review_page_url}")

        # Navigate back to the search page
        driver.get("https://www.google.com/maps")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "searchboxinput")))

    except Exception as e:
        print(f"Error processing {search_query}: {e}")
        # Navigate back to the search page even if an error occurs
        driver.get("https://www.google.com/maps")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "searchboxinput")))
        continue

# Save results to the data folder
results_df = pd.DataFrame(results)
results_df.to_csv("../data/scraped_google_maps_urls.csv", index=False)
print("Results saved to ../data/scraped_google_maps_urls.csv")

# Close the browser
driver.quit()