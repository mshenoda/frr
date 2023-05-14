import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import re

def plot_image(url):
    # Remove the #gallery part from the URL
    url = re.sub(r'#.*$', '', url)

    # Send a GET request to the URL
    response = requests.get(url)
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all image tags in the HTML
    img_tags = soup.find_all('img')
    
    if len(img_tags) >= 2:
        # Extract the second image URL
        img_url = img_tags[1]['src']
        
        # Send a GET request to the image URL
        img_response = requests.get(img_url)
        
        # Read the image data
        img_data = BytesIO(img_response.content)
        
        # Open the image using PIL
        img = Image.open(img_data)
        
        # Display the image using Matplotlib
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print("")

def create_recipe_url(recipe_id, recipe_name):
    # Remove leading/trailing white spaces from the recipe name
    recipe_name = recipe_name.strip()

    # Split the recipe name into individual words
    words = recipe_name.split()

    # Join the words using a single space as the separator
    formatted_name = ' '.join(words)

    # Replace remaining white spaces in the recipe name with dashes
    formatted_name = formatted_name.replace(' ', '-')
    
    # Create the URL by concatenating the recipe ID and formatted recipe name
    url = f"https://www.food.com/recipe/{formatted_name}-{recipe_id}"
    
    return url

def plot_recipe(id, name):
    plot_image(create_recipe_url(id, name)+"#gallery")