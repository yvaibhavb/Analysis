import requests
from bs4 import BeautifulSoup

def crawl(url):
  # make a request to the website
  response = requests.get(url)
  # parse the html content
  soup = BeautifulSoup(response.text, 'html.parser')
  # find all the links on the page
  links = soup.find_all('a')
  # print the links
  for link in links:
    print(link.get('href'))

# start crawling from the homepage
crawl('https://www.google.com')
