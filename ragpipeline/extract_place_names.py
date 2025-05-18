import csv

# Path to your CSV file
csv_file = 'review_summaries.csv'

# List to store place names
place_names = []

# Read the CSV file
with open(csv_file, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    # Skip the header row
    next(csv_reader)
    
    # Extract place names
    for row in csv_reader:
        if row:  # Check if row is not empty
            place_name = row[0].strip()
            # Some place names have quotes, remove them
            place_name = place_name.replace('"', '')
            place_names.append(place_name)

# Print the list of place names
print("List of place names:")
for place in place_names:
    print(place)

# Optionally, save the list to a new file
with open('place_names.txt', 'w', encoding='utf-8') as file:
    for place in place_names:
        file.write(place + "\n")

print(f"\nTotal places: {len(place_names)}")
print("Place names have been saved to 'place_names.txt'")
