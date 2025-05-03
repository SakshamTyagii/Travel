from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
import time
import pandas as pd
import random
import os
import re
from datetime import datetime
import urllib.parse

def load_urls_from_csv(filename="../data/scraped_google_maps_urls.csv"):
    """Load the previously scraped location URLs from CSV"""
    try:
        df = pd.read_csv(filename, encoding='utf-8')
        # Ensure required columns exist
        if 'Location' not in df.columns or 'URL' not in df.columns:
            raise ValueError("CSV must contain 'Location' and 'URL' columns")
        # Convert to list of dictionaries with expected column names
        return df[['Location', 'URL']].rename(columns={
            'Location': 'place_name',
            'URL': 'place_url'
        }).to_dict('records')
    except FileNotFoundError:
        print(f"⚠️ File not found: {filename}")
        print("Make sure the CSV file exists.")
        return []
    except Exception as e:
        print(f"⚠️ Error loading CSV: {str(e)}")
        return []

def clean_place_name(name):
    """Clean the place name for consistency"""
    # For airport terminals, use the main airport name
    if any(term in name.lower() for term in ["terminal", "t1", "t2", "t3"]) and any(term in name.lower() for term in ["indira", "igi", "gandhi", "airport"]):
        return "Indira Gandhi International Airport"
    
    # For railway stations and bus terminals
    if any(term in name.lower() for term in ["railway", "station", "isbt", "bus terminal", "bus stand"]):
        return name
    
    return name

def get_parent_location_url(place_name, place_url):
    """Get the parent location URL for places like airport terminals"""
    # For airport terminals, use the main airport URL
    if any(term in place_name.lower() for term in ["terminal", "t1", "t2", "t3"]) and any(term in name.lower() for term in ["indira", "igi", "gandhi", "airport"]):
        return "https://www.google.com/maps/place/Indira+Gandhi+International+Airport/reviews"
    
    # For railway stations
    if any(station in place_name.lower() for station in ["new delhi", "delhi", "anand vihar", "hazrat nizamuddin", "old delhi"]) and "railway" in place_name.lower():
        station_name = place_name.replace("Railway Station", "").strip()
        return f"https://www.google.com/maps/place/{urllib.parse.quote(station_name)}+Railway+Station/reviews"
    
    # For ISBTs
    if "isbt" in place_name.lower():
        isbt_name = place_name.replace("ISBT", "").strip()
        return f"https://www.google.com/maps/place/{urllib.parse.quote(isbt_name)}+ISBT/reviews"
        
    return None

def scrape_all_reviews(driver, place_name, target_reviews):
    """Scrape reviews without filtering by rating"""
    return collect_visible_reviews(driver, place_name, None, target_reviews)

def collect_visible_reviews(driver, place_name, star_filter, target_reviews):
    """Collect visible reviews with robust handling"""
    reviews = []
    scroll_attempts = 0
    max_scroll_attempts = 50
    last_review_count = 0
    no_new_reviews_count = 0
    
    review_selectors = [
        "//div[contains(@class, 'fontBodyMedium')][.//div[string-length(text()) > 5]]",
        "//div[@data-review-id]",
        "//div[contains(@class, 'jftiEf')]",
        "//div[contains(@class, 'MyEned')]/ancestor::div[contains(@class, 'fontBodyMedium')]",
        "//div[contains(@class, 'wiI7pd')]/ancestor::div[contains(@class, 'fontBodyMedium')]",
        "//div[contains(@class, 'jJc9Ad')]"
    ]
    
    more_reviews_selectors = [
        "//button[contains(., 'More reviews')]",
        "//button[contains(@aria-label, 'More reviews')]",
        "//button[contains(@jsaction, 'pane.review.moreReviews')]",
        "//div[contains(., 'More reviews') and @role='button']",
        "//button[contains(@class, 'more-reviews')]"
    ]
    
    while len(reviews) < target_reviews and scroll_attempts < max_scroll_attempts and no_new_reviews_count < 5:
        try:
            current_reviews = []
            for selector in review_selectors:
                elements = driver.find_elements(By.XPATH, selector)
                if elements:
                    current_reviews = elements
                    break
            
            if not current_reviews:
                print("  ⚠️ No review elements found with any selector")
                scroll_attempts += 1
                continue
            
            new_reviews_found = 0
            
            for review_elem in current_reviews:
                try:
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", review_elem)
                    time.sleep(0.1)
                    
                    review_id = review_elem.get_attribute('data-review-id')
                    if not review_id:
                        try:
                            review_text_elements = review_elem.find_elements(By.XPATH, ".//span[@class='wiI7pd'] | .//div[@class='MyEned'] | .//div[contains(@class, 'review-text')]")
                            if review_text_elements:
                                snippet = review_text_elements[0].text[:50]
                                review_id = f"text_{hash(snippet)}"
                            else:
                                review_id = f"html_{hash(review_elem.get_attribute('outerHTML')[:100])}"
                        except:
                            review_id = f"elem_{len(reviews)}"
                    
                    if any(r.get('review_id') == review_id for r in reviews):
                        continue
                    
                    review_data = {
                        'review_id': review_id,
                        'place_name': place_name
                    }
                    
                    # Extract rating
                    try:
                        rating_elements = review_elem.find_elements(By.XPATH, 
                            ".//span[contains(@aria-label, 'star')] | .//span[contains(@aria-label, 'rating')] | .//div[contains(@aria-label, 'star')] | .//div[contains(@aria-label, 'rating')] | .//span[contains(@class, 'KdvmLc')] | .//g-review-stars"
                        )
                        
                        rating = None
                        for rating_elem in rating_elements:
                            aria_label = rating_elem.get_attribute('aria-label')
                            if aria_label:
                                match = re.search(r'(\d+(?:\.\d+)?)', aria_label)
                                if match:
                                    rating = int(float(match.group(1)))
                                    break
                        
                        if rating is None:
                            star_elements = review_elem.find_elements(By.XPATH, ".//img[contains(@src, 'star')] | .//span[contains(@class, 'star')]")
                            if star_elements:
                                filled_stars = sum(1 for s in star_elements if 'filled' in s.get_attribute('src').lower() or 'filled' in s.get_attribute('class').lower())
                                if filled_stars > 0:
                                    rating = filled_stars
                        
                        review_data['rating'] = rating if rating is not None else 0
                    except Exception:
                        review_data['rating'] = 0
                    
                    if star_filter is not None and review_data['rating'] != star_filter:
                        continue
                    
                    # Extract reviewer name
                    try:
                        name_elements = review_elem.find_elements(By.XPATH, 
                            ".//div[contains(@class, 'voVFBc')] | .//div[contains(@class, 'd4r55')] | .//div[contains(@class, 'user-name')] | .//button[contains(@jsaction, 'pane.review')]"
                        )
                        review_data['reviewer_name'] = name_elements[0].text if name_elements else "Anonymous"
                    except Exception:
                        review_data['reviewer_name'] = "Anonymous"
                    
                    # Extract review text
                    try:
                        text_elements = review_elem.find_elements(By.XPATH, 
                            ".//span[@class='wiI7pd'] | .//div[@class='MyEned'] | .//div[contains(@class, 'review-text')] | .//div[contains(@class, 'review-full-text')] | .//span[contains(@jsaction, 'pane.review')]"
                        )
                        review_text = ""
                        if text_elements:
                            review_text = text_elements[0].text
                            more_buttons = review_elem.find_elements(By.XPATH, 
                                ".//button[contains(., 'More')] | .//button[contains(@jsaction, 'pane.review.expandReview')] | .//button[contains(@class, 'expand')]"
                            )
                            if more_buttons:
                                try:
                                    more_button = more_buttons[0]
                                    if more_button.is_displayed():
                                        driver.execute_script("arguments[0].click();", more_button)
                                        time.sleep(random.uniform(0.5, 1))
                                        expanded_text_elements = review_elem.find_elements(By.XPATH, 
                                            ".//span[@class='wiI7pd'] | .//div[@class='MyEned'] | .//div[contains(@class, 'review-text')] | .//div[contains(@class, 'review-full-text')]"
                                        )
                                        if expanded_text_elements:
                                            review_text = expanded_text_elements[0].text
                                except Exception:
                                    pass
                        review_data['review_text'] = review_text
                    except Exception:
                        review_data['review_text'] = ""
                    
                    # Extract date
                    try:
                        date_elements = review_elem.find_elements(By.XPATH, 
                            ".//span[contains(@class, 'rsqaWe')] | .//span[contains(@class, 'date')] | .//span[contains(@class, 'review-date')] | .//div[contains(@class, 'date')]"
                        )
                        review_data['date'] = date_elements[0].text if date_elements else "Unknown date"
                    except Exception:
                        review_data['date'] = "Unknown date"
                    
                    if review_data['review_text'] and len(review_data['review_text'].strip()) > 5:
                        reviews.append(review_data)
                        new_reviews_found += 1
                except StaleElementReferenceException:
                    continue
                except Exception as e:
                    print(f"  ⚠️ Error processing review: {str(e)[:50]}...")
                    continue
            
            # Try clicking 'More reviews' buttons
            more_reviews_clicked = False
            for selector in more_reviews_selectors:
                try:
                    more_buttons = driver.find_elements(By.XPATH, selector)
                    for button in more_buttons:
                        if button.is_displayed():
                            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
                            driver.execute_script("arguments[0].click();", button)
                            print("  ✓ Clicked 'More reviews' button")
                            time.sleep(random.uniform(2, 4))
                            more_reviews_clicked = True
                            break
                    if more_reviews_clicked:
                        break
                except Exception:
                    continue
            
            # Scroll to load more reviews
            try:
                scroll_element = driver.find_element(By.XPATH, 
                    "//div[contains(@role, 'main')] | //div[contains(@class, 'review-dialog-list')] | //div[contains(@data-review-id)] | //div[contains(@class, 'section-scrollbox')]"
                )
                driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_element)
            except:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            if len(reviews) == last_review_count and not more_reviews_clicked:
                no_new_reviews_count += 1
            else:
                no_new_reviews_count = 0
                
            last_review_count = len(reviews)
            time.sleep(random.uniform(1, 2))
            scroll_attempts += 1
            
            if len(reviews) % 25 == 0 and len(reviews) > 0:
                print(f"  📊 Collected {len(reviews)} reviews ({scroll_attempts}/{max_scroll_attempts} scroll attempts)")
                
        except Exception as e:
            print(f"  ⚠️ Error during review collection: {str(e)[:100]}...")
            scroll_attempts += 1
    
    if len(reviews) > 0:
        print(f"  ✓ Successfully collected {len(reviews)} reviews after {scroll_attempts} scroll attempts")
    else:
        print(f"  ⚠️ No reviews collected after {scroll_attempts} scroll attempts")
        
    return reviews

def scrape_by_star_rating(driver, place_name, star_rating, target_reviews):
    """Scrape reviews filtered by star rating"""
    try:
        # Try to open the star filter menu
        filter_buttons = driver.find_elements(By.XPATH, 
            "//button[contains(@aria-label, 'Filter')] | //button[contains(., 'Filter')] | //div[contains(@role, 'button')][contains(., 'Filter')]"
        )
        if filter_buttons:
            driver.execute_script("arguments[0].click();", filter_buttons[0])
            time.sleep(random.uniform(1, 2))
            
            # Select the star rating
            star_option = driver.find_elements(By.XPATH, 
                f"//div[contains(., '{star_rating} star')] | //div[contains(., '{star_rating} stars')] | //input[@value='{star_rating}']"
            )
            if star_option:
                driver.execute_script("arguments[0].click();", star_option[0])
                time.sleep(random.uniform(2, 3))
                
                # Collect filtered reviews
                return collect_visible_reviews(driver, place_name, star_rating, target_reviews)
        
        print(f"  ⚠️ Could not apply {star_rating}-star filter")
        return []
    except Exception as e:
        print(f"  ⚠️ Error filtering by {star_rating} stars: {str(e)[:100]}...")
        return []

def save_progress(all_reviews, all_places_data, output_dir):
    """Save all reviews and metadata to single CSV files"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all reviews to a single CSV
    reviews_df = pd.DataFrame(all_reviews)
    reviews_file = os.path.join(output_dir, "google_maps_reviews.csv")
    reviews_df.to_csv(reviews_file, index=False, encoding='utf-8')
    print(f"  💾 Saved {len(all_reviews)} reviews to {reviews_file}")
    
    # Save all place metadata to a single CSV
    places_df = pd.DataFrame(all_places_data)
    places_file = os.path.join(output_dir, "google_maps_places_metadata.csv")
    places_df.to_csv(places_file, index=False, encoding='utf-8')
    print(f"  💾 Saved metadata for {len(all_places_data)} places to {places_file}")

def chunk_reviews(reviews, max_chars=4000):  # Reduced from 8000 to 4000
    """Split reviews into smaller chunks to avoid memory issues"""
    reviews = reviews[:max_chars] if len(reviews) > max_chars else reviews
    return reviews

def scrape_reviews(urls_data, output_dir="../data", min_reviews=150, max_reviews=500, star_filter=True):
    """
    Scrape Google Maps reviews from provided review section URLs and save to single CSV files
    
    Args:
        urls_data: List of dictionaries with place_name and place_url
        output_dir: Directory to save CSV files
        min_reviews: Minimum number of reviews to scrape per place
        max_reviews: Maximum number of reviews to scrape per place
        star_filter: Whether to filter by star rating (5,4,3 stars)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Chrome driver
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    service = Service()
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Anti-detection
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    all_reviews = []
    all_places_data = []
    
    for i, place_data in enumerate(urls_data):
        place_name = place_data["place_name"]
        place_url = place_data["place_url"]
        
        print(f"\n[{i+1}/{len(urls_data)}] 🔍 Scraping reviews for: {place_name}")
        
        place_reviews = []
        place_info = {
            "place_name": place_name,
            "place_url": place_url,
            "reviews_count": 0,
            "scraped_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            # STAGE 1: Try direct URL
            print(f"  🔍 STAGE 1: Accessing reviews directly at {place_url}")
            driver.get(place_url)
            time.sleep(random.uniform(4, 6))
            
            review_found = False
            # Check if reviews are visible
            if len(driver.find_elements(By.XPATH, "//div[@data-review-id] | //div[contains(@class, 'fontBodyMedium')][.//div[contains(@class, 'wiI7pd')]]")) > 0:
                print("  ✓ Reviews found at direct URL")
                review_found = True
            
            # STAGE 2: Try parent location if no reviews found
            if not review_found:
                print("  🔍 STAGE 2: Trying parent location...")
                parent_url = get_parent_location_url(place_name, place_url)
                if parent_url:
                    print(f"  ⚠️ Trying parent location URL: {parent_url}")
                    driver.get(parent_url)
                    time.sleep(random.uniform(4, 6))
                    
                    if len(driver.find_elements(By.XPATH, "//div[@data-review-id] | //div[contains(@class, 'fontBodyMedium')][.//div[contains(@class, 'wiI7pd')]]")) > 0:
                        print("  ✓ Reviews found at parent location")
                        review_found = True
            
            # If reviews are found
            if review_found:
                # Try sorting reviews for better quality
                sorting_attempted = False
                try:
                    print("  🔄 Trying to sort reviews...")
                    sort_button_selectors = [
                        "//button[contains(@aria-label, 'Sort')]",
                        "//button[contains(., 'Sort')]",
                        "//div[contains(@role, 'button')][contains(., 'Sort')]",
                        "//button[contains(@jsaction, 'pane.review.sort')]"
                    ]
                    
                    sort_button = None
                    for selector in sort_button_selectors:
                        buttons = driver.find_elements(By.XPATH, selector)
                        for button in buttons:
                            if button.is_displayed():
                                sort_button = button
                                break
                        if sort_button:
                            break
                    
                    if sort_button:
                        driver.execute_script("arguments[0].click();", sort_button)
                        time.sleep(random.uniform(1.5, 2.5))
                        
                        sort_options = ["Newest", "Most relevant", "Highest rating"]
                        for option in sort_options:
                            try:
                                option_selector = f"//div[contains(., '{option}') and @role='menuitem'] | //li[contains(., '{option}')]"
                                option_elements = driver.find_elements(By.XPATH, option_selector)
                                if option_elements:
                                    for elem in option_elements:
                                        if option.lower() in elem.text.lower() and elem.is_displayed():
                                            driver.execute_script("arguments[0].click();", elem)
                                            print(f"  ✓ Applied sort: {option}")
                                            time.sleep(random.uniform(2, 3))
                                            sorting_attempted = True
                                            break
                                    if sorting_attempted:
                                        break
                            except Exception:
                                continue
                except Exception as e:
                    print(f"  ⚠️ Error while sorting: {str(e)[:100]}")
                
                # Collect all reviews
                print("  📥 Collecting reviews...")
                all_reviews_list = scrape_all_reviews(driver, place_name, min_reviews)
                
                if all_reviews_list:
                    place_reviews.extend(all_reviews_list)
                    print(f"  ✓ Collected {len(all_reviews_list)} reviews")
                
                # If star filter is enabled and not enough reviews
                if star_filter and len(place_reviews) < min_reviews:
                    print("  🌟 Attempting to collect reviews by star rating...")
                    star_targets = [5, 4, 3]
                    reviews_per_star = min_reviews // len(star_targets)
                    for star in star_targets:
                        star_reviews = scrape_by_star_rating(driver, place_name, star, reviews_per_star)
                        if star_reviews:
                            place_reviews.extend(star_reviews)
                            print(f"  ✓ Collected {len(star_reviews)} {star}-star reviews")
            
            else:
                print(f"  ❌ No reviews found for {place_name}")
            
            # Limit to max_reviews
            if len(place_reviews) > max_reviews:
                place_reviews = place_reviews[:max_reviews]
            
            # Update place info
            place_info["reviews_count"] = len(place_reviews)
            
            # Add to collections
            all_reviews.extend(place_reviews)
            all_places_data.append(place_info)
            
            print(f"  ✅ Scraped {len(place_reviews)} reviews for {place_name}")
            
            # Save progress to single CSV files
            save_progress(all_reviews, all_places_data, output_dir)
            
        except Exception as e:
            print(f"  ❌ Error processing {place_name}: {str(e)[:100]}...")
        
        # Random delay
        delay = random.uniform(5, 10)
        print(f"  ⏱️ Waiting {delay:.1f} seconds before next place...")
        time.sleep(delay)
    
    # Final save to single CSV files
    save_progress(all_reviews, all_places_data, output_dir)
    
    driver.quit()
    print(f"\n✅ Scraping completed! Total reviews: {len(all_reviews)}")
    return all_reviews

if __name__ == "__main__":
    print("📱 Google Maps Review Scraper")
    print("-----------------------------")
    
    # Load URLs from CSV
    urls_data = load_urls_from_csv()
    print(f"📍 Loaded {len(urls_data)} places to scrape reviews for.")
    
    if not urls_data:
        print("❌ No URLs to scrape. Exiting...")
        exit()
    
    # Get user input
    start_from = input("Start scraping from which location number? (default: 0): ").strip()
    start_from = int(start_from) if start_from.isdigit() else 0
    
    # Slice the URLs data to start from specified location
    if start_from > 0:
        urls_data = urls_data[start_from:]
        print(f"➡️ Starting from location #{start_from}")
    
    min_reviews = input("Minimum reviews per place (default 150): ")
    min_reviews = int(min_reviews) if min_reviews.strip().isdigit() else 150
    
    max_reviews = input("Maximum reviews per place (default 500): ")
    max_reviews = int(max_reviews) if max_reviews.strip().isdigit() else 500
    
    print("🚀 Starting review scraper...")
    
    # Start scraping and save to single CSV files
    scraped_reviews = scrape_reviews(
        urls_data=urls_data,
        min_reviews=min_reviews,
        max_reviews=max_reviews,
        star_filter=True
    )
    
    print(f"🎉 Done! Scraped {len(scraped_reviews)} total reviews.")