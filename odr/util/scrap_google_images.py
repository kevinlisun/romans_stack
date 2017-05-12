#!/usr/bin/env python

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os
import urllib2

searchterm = raw_input("searchterm: ")
url = "https://www.google.co.in/search?q="+searchterm+"&source=lnms&tbm=isch"
browser = webdriver.Firefox(executable_path='./geckodriver')
browser.get(url)
header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
counter = 0
succounter = 0

folder = "/home/kevin/Pictures" 

if not os.path.exists(folder):
    os.mkdir(folder)

folder = folder + "/" + searchterm

if not os.path.exists(folder):
    os.mkdir(folder)

for _ in range(1000):
    browser.execute_script("window.scrollBy(0,10000)")

for x in browser.find_elements_by_xpath("//div[@class='rg_meta']"):
    counter = counter + 1
    print "Total Count:", counter
    print "Succsessful Count:", succounter
    print "URL:",json.loads(x.get_attribute('innerHTML'))["ou"]

    img = json.loads(x.get_attribute('innerHTML'))["ou"]
    imgtype = json.loads(x.get_attribute('innerHTML'))["ity"]
    try:
        req = urllib2.Request(img, headers={'User-Agent': header})
        raw_img = urllib2.urlopen(req).read()
        File = open(os.path.join(folder , searchterm + "_" + str(counter) + "." + imgtype), "wb")
        File.write(raw_img)
        File.close()
        succounter = succounter + 1
    except:
            print "can't get img"

print succounter, "pictures succesfully downloaded"
browser.close()

