import string
import csv
import re
import json
from collections import defaultdict
import pandas as pd
import numpy as np
# from google.colab import drive
from platform import python_version
import os
import json
# final_arr=[]
# drive.mount('/content/gdrive')
# os.chdir('/content/gdrive/My Drive/IR/HW4/')
import requests
from bs4 import BeautifulSoup
import re
# URL of the webpage to scrape
url = "https://mtsamples.com/"
start_word = "Description:"
end_word = ["Keywords:"]
file_path = "output.txt"
if os.path.exists(file_path):
    # Remove the file
    os.remove(file_path)
    print(f"File '{file_path}' has been deleted.")

ans_arr=[]

# Send a GET request to the webpage and get its HTML content
response = requests.get(url)
html_content = response.content

# Parse the HTML content using Beautiful Soup
soup = BeautifulSoup(html_content, "html.parser")

# Find all the links on the webpage
links = soup.find_all("a")

# Loop through each link on the webpage
for link in links:
    # Get the URL of the link
    link_url = link.get("href")
    if "/site/pages/browse.asp?type=" in link_url:
        link_url=url + link_url
       
        
        
    
    # Send a GET request to the link URL and get its HTML content
        link_response = requests.get(link_url)
        link_html_content = link_response.content
        
        # Parse the HTML content of the link using Beautiful Soup
        link_soup = BeautifulSoup(link_html_content, "html.parser")
        
        # Find all the links on the link page
        link_links = link_soup.find_all("a")
        final_arr=[]
        # Loop through each link on the link page and print its URL
        for link_link in link_links:
            link_link_url = url+link_link.get("href")
            if "/site/pages/browse.asp?type=" and "&Sample=" in link_link_url:
                
                
                response= requests.get(link_link_url)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "html.parser")

                    text_content = soup.get_text()

                    line=text_content.split('\n')
                    
                    clean_list = list(filter(None, line))
                    # print(clean_list[154:174])

                    final_arr.append(clean_list[154:174])
                    
                    # for i, elem in enumerate(clean_list):
                    #   if "Sample Name:\xa0" in elem:
                    #       print("Index of element containing 'Sample Name:':", i)
                    #       break

                    # for i, elem in enumerate(clean_list):
                    #   if "Keywords" in elem:
                    #       print("Index of element containing 'Sample Name:':", i)
                    #       break

                    # for i, elem in enumerate(clean_list):
                    #   if "NOTE" in elem:
                    #       print("Index of element containing 'Sample Name:':", i)
                    #       break
                    print(final_arr)

                    with open("output.txt", "a") as f:
                    
                      f.write("%s\n" % clean_list[154:174])
                      f.write("\n")
                    
                    


                    
                    
                    
                    