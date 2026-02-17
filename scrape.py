import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from tqdm import tqdm  # Shows a progress bar
import os
from datetime import datetime
import glob

# ==========================================
# 1. DEFINE EXTRACTION LOGIC FOR CAR DATA
# ==========================================
def extract_ad_data(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        data = {
            'brand': '',
            'model': '',
            'trim': '',
            'manufacture_year': '',
            'condition': '',
            'transmission': '',
            'body_type': '',
            'fuel_type': '',
            'engine_capacity': '',
            'mileage': '',
            'price': ''
        }
        
        # 1. EXTRACT PRICE
        price_tag = soup.find('div', class_=lambda x: x and 'amount--' in x)
        if price_tag:
            # Cleans "Rs 8,975,000" -> "8975000"
            data['price'] = price_tag.get_text(strip=True).replace('Rs', '').replace(',', '').strip()
        
        # 2. EXTRACT ALL CAR DETAILS
        # Find all labels (div.label--3oVZK) and their values (div.value--1lKHt)
        labels = soup.find_all('div', class_=lambda x: x and 'label--' in x)
        
        for label in labels:
            key = label.get_text(strip=True).replace(':', '').strip()
            
            # The value is usually the NEXT sibling div
            value_div = label.find_next_sibling('div')
            
            if value_div:
                value = value_div.get_text(strip=True)
                
                # Map to our standard column names
                if "Brand" in key:
                    data['brand'] = value
                elif "Model" in key:
                    data['model'] = value
                elif "Trim" in key or "Edition" in key:
                    data['trim'] = value
                elif "Year of Manufacture" in key:
                    data['manufacture_year'] = value
                elif "Condition" in key:
                    data['condition'] = value
                elif "Transmission" in key:
                    data['transmission'] = value
                elif "Body type" in key:
                    data['body_type'] = value
                elif "Fuel type" in key:
                    data['fuel_type'] = value
                elif "Engine capacity" in key:
                    data['engine_capacity'] = value
                elif "Mileage" in key:
                    data['mileage'] = value

        return data

    except Exception as e:
        return None

# ==========================================
# 2. PROCESS EACH BRAND FILE
# ==========================================

def process_brand_file(txt_file_path):
    """Process a single brand txt file and create corresponding CSV"""
    
    # Extract brand name from filename (e.g., 'toyota.txt' -> 'toyota')
    brand_name = os.path.basename(txt_file_path).replace('.txt', '')
    
    print(f"\n{'='*50}")
    print(f"Processing: {brand_name.upper()}")
    print(f"{'='*50}")
    
    # 1. LOAD & DEDUPLICATE LINKS
    with open(txt_file_path, 'r') as f:
        raw_links = f.readlines()
    
    # Clean whitespace and remove duplicates
    unique_links = list(set([link.strip() for link in raw_links if "ikman.lk" in link]))
    
    print(f"Original Count: {len(raw_links)}")
    print(f"Unique Links to Scrape: {len(unique_links)}")
    print("-" * 50)
    
    # 2. SETUP OUTPUT FILES
    csv_filename = f"data/car-details/{brand_name}_cars.csv"
    failed_links_filename = f"data/failed-links/{brand_name}_failed_links.txt"
    
    data_buffer = []
    failed_links = []
    
    # 3. SCRAPE EACH LINK
    print(f"Scraping {brand_name} ads...")
    
    for i, link in tqdm(enumerate(unique_links), total=len(unique_links)):
        
        # Extract data
        row = extract_ad_data(link)
        
        if row:
            data_buffer.append(row)
        else:
            # Track failed link
            failed_links.append(link)
        
        # SAVE EVERY 50 LINKS (Checkpointing)
        if len(data_buffer) >= 50:
            df_chunk = pd.DataFrame(data_buffer)
            
            # If file doesn't exist, write header. If it does, append without header.
            if not os.path.isfile(csv_filename):
                df_chunk.to_csv(csv_filename, index=False)
            else:
                df_chunk.to_csv(csv_filename, mode='a', header=False, index=False)
            
            data_buffer = [] # Clear buffer
            
        # Rate Limiting (Crucial to avoid ban)
        time.sleep(random.uniform(0.5, 1.5))
    
    # 4. SAVE REMAINING DATA
    if data_buffer:
        df_chunk = pd.DataFrame(data_buffer)
        if not os.path.isfile(csv_filename):
            df_chunk.to_csv(csv_filename, index=False)
        else:
            df_chunk.to_csv(csv_filename, mode='a', header=False, index=False)
    
    print(f"\n✓ Success! Data saved to {csv_filename}")
    
    # 5. SAVE FAILED LINKS
    if failed_links:
        with open(failed_links_filename, 'w') as f:
            f.write("\n".join(failed_links))
        print(f"✗ Failed links saved to {failed_links_filename} ({len(failed_links)} links)")
    else:
        print("✓ No failed links!")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Find all txt files in the car-ad-links folder
    txt_files = glob.glob("data/car-ad-links/*.txt")
    
    if not txt_files:
        print("No txt files found in data/car-ad-links/")
        exit(1)
    
    print(f"Found {len(txt_files)} brand files to process:")
    for file in txt_files:
        print(f"  - {os.path.basename(file)}")
    
    # Process each brand file sequentially
    for txt_file in txt_files:
        process_brand_file(txt_file)
    
    print(f"\n{'='*50}")
    print("ALL BRANDS PROCESSED SUCCESSFULLY!")
    print(f"{'='*50}")