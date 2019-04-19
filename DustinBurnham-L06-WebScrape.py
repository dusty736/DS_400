#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 20:18:55 2019

@author: dusty

Dustin Burnham
Data Science 400
2/12/2019
Lesson 6 Assignment: Web Scrape

Steps:
1. Import statements for necessary package(s)
2. Read in an html page from a freely and easily available source on the internet. The html page must contain at least 3 links.
3. Write code to tally the number of links in the html page.
4. Use print to present the tally

Summary:
    In this file I will be pulling the html code from the front page of 
    meetup.com using the requests package.  From here I will use beautiful
    soup to parse the data and find all 'a' tags which indicate a url link.
    The soup.find_all() function returns an array of all links.  The length
    of this array will be the number of links, which is 82.  I print this
    tally as the file output.
"""

# Import requests and beautiful soup packages
import requests
from bs4 import BeautifulSoup 

# Selected the URL form the meetup.com front page
url = "https://www.meetup.com/"

# Pull down the html and stor it in a request object
response = requests.get(url)

# Content from the website
content = response.content

# Convert the data into using the 'lxml' tag.  
soup = BeautifulSoup(content, 'lxml')

# Look for the tag 'a', which indicates a url link.  The result
# will be an array of url's.
all_a_https = soup.find_all("a")   

# The length of the url array will give us the amount of links on the 
# meetup.com front page.
tally = len(all_a_https)

# Print the number of links.
print("Links Tally:", tally)