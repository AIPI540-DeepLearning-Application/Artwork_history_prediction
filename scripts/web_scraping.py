from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import csv
import time

class artwork_scraper:
    '''
    This class is used to scrape the artwork data from the National Gallery of Art website
    '''
    
    def main(self):
        # Open the browser
        driver = webdriver.Chrome()
        
        max_mum = 57679
        csv_filename = 'arts_info.csv'

        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    'title',
                    'year',
                    'link'
                ]
            )
            for i in range(20700, 50000, 5):
                url = 'https://www.nga.gov/collection/art-object-page.' + str(i) + '.html'
                driver.get(url)
                
                # scrape the image
                try:
                    xpath = '/html/body/div[2]/div[1]/div[2]/div[1]/div[2]/div[1]/div/div/div[1]/div[1]/ul[1]/li/span/img'
                    img_url = (
                        WebDriverWait(driver, 2)
                        .until(EC.presence_of_element_located((By.XPATH, xpath)))
                    )
                    img_url = img_url.get_attribute('src')
                except:
                    img_url = ''
                
                # scrape the year of the artwork
                try:
                    xpath = '/html/body/div[2]/div[1]/div[2]/div[1]/div[2]/div[1]/div/div/div[2]/h1/span'
                    year = (
                        WebDriverWait(driver, 2)
                        .until(EC.presence_of_element_located((By.XPATH, xpath)))
                    )
                except:
                    year = ''
                
                # scrape the title of the artwork
                try:
                    xpath = '/html/body/div[2]/div[1]/div[2]/div[1]/div[2]/div[1]/div/div/div[2]/h1/em'
                    title = (
                        WebDriverWait(driver, 2)
                        .until(EC.presence_of_element_located((By.XPATH, xpath)))
                    )
                except:
                    title = ''
                    
                if title and year and img_url:
                    print(title.text, year.text, img_url)
                    writer.writerow(
                        [
                            title.text,
                            year.text,
                            img_url 
                        ]
                    )
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")