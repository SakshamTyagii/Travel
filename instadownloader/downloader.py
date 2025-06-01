import csv
import os
import requests
from urllib.parse import quote

CSV_FILE = "urls.csv"

def read_urls_and_ids(file_path):
    with open(file_path, newline='') as file:
        reader = csv.DictReader(file)
        # Return both ID and Link for each row
        return [(row['ID'], row['Link']) for row in reader if row.get('Link') and row.get('ID')]

def download_video(url, video_id, index):
    try:
        # Make request to local API
        api_url = f"https://instagram-reels-downloader-tau.vercel.app/api/video?postUrl={url}"
        
        # Create videos directory if it doesn't exist
        os.makedirs('videos', exist_ok=True)
        
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        print(f"[{index}] Requesting video from local API: {url}")
        response = requests.get(api_url, headers=headers)
        
        if response.status_code == 200:
            try:
                video_data = response.json()
                # print(f"[{index}] Response received: {video_data}")  # Debug print
                
                # Check for nested videoUrl in data
                if video_data.get('status') == 'success' and video_data.get('data', {}).get('videoUrl'):
                    video_url = video_data['data']['videoUrl']
                    # Download the actual video data
                    video_response = requests.get(video_url)
                    if video_response.status_code == 200:
                        filename = os.path.join('videos', f'{video_id}.mp4')
                        with open(filename, 'wb') as f:
                            f.write(video_response.content)
                        print(f"[{index}] Successfully downloaded: {url} as {video_id}.mp4")
                    else:
                        print(f"[{index}] Failed to download video content: HTTP {video_response.status_code}")
                else:
                    print(f"[{index}] No video URL in response or invalid response format")
            except ValueError as e:
                print(f"[{index}] Failed to parse JSON response: {e}")
        else:
            print(f"[{index}] Failed to get video info: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"[{index}] Error: {str(e)}")
        return None

def main():
    print("Starting Instagram Video Downloader...")
    url_data = read_urls_and_ids(CSV_FILE)
    if not url_data:
        print("No URLs found in CSV.")
        return

    for i, (video_id, url) in enumerate(url_data, start=1):
        download_video(url, video_id, i)

    print("All done!")

if __name__ == "__main__":
    main()
