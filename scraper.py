#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup



#load in data
def getSoup(url):
	page = requests.get(url).text
	soup = BeautifulSoup(page, 'html.parser')
	return soup

if __name__ == '__main__':
	getSoup()