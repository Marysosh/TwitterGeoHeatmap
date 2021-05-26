from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
import sys

import json
import unittest, time, re
from bs4 import BeautifulSoup as bs
from dateutil import parser
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from math import ceil

from array import *
import csv

homeUrl = 'https://twitter.com/home'
accountNickname = 'allbirds'
followersNumber = 28706

#starting webdriver
driver = webdriver.Firefox()
driver.base_url = "https://twitter.com/login"
driver.get(driver.base_url)

WebDriverWait(driver, 400).until(EC.url_to_be(homeUrl))



#####################################################################
########                                                     ########
########              Followers list parsing                 ########
########                                                     ########
#####################################################################



followersUrl = "https://twitter.com/"+accountNickname+"/followers"
driver.get(followersUrl)

print('Starting parsing followers...')

nicknamesArray = []
iter = ceil(followersNumber / 4.5)
# iter = 3
print('Total number of iters is ' + str(iter))
for i in range(1,iter):
    html_source = driver.page_source
    sourcedata= html_source.encode('utf-8')
    soup=bs(sourcedata, features="html.parser")

    elements = [x.text.strip() for x in soup.body.findAll('span', 'css-901oao css-16my406 r-poiln3 r-bcqeeo r-qvutc0')]
    elements = filter(lambda s: len(s) and s[0] == '@',elements)
    elements = map(lambda s: s[1:],elements)
    # print(type(elements))

    for element in elements:
        if element not in nicknamesArray:
            nicknamesArray.append(element)


    # if i % 20 == 0:
    #     print(list(nicknamesArray))


    # nicknamesArray.extend(list(elements))
    time.sleep(3)
    # driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    scrollValue = 800 * i
    print(scrollValue)

    driver.execute_script("window.scrollTo(0, " + str(scrollValue) + ");")
    print('Iter ' + str(i))
    print('Total followers ' + str(len(nicknamesArray)))

nicknamesArray = list(set(nicknamesArray))
df_nicks_data = {}
df_nicks_data["NICKNAME"] = nicknamesArray
df_nicks = pd.DataFrame(data=df_nicks_data)
df_nicks.to_csv('nicknames.csv', sep='	', index=False, header=False)

print('Total followers ' + str(len(nicknamesArray)))
print('Followers list ' + str(nicknamesArray))

nickname = ''


#####################################################################
########                                                     ########
########              Parse followers' tweets                ########
########                                                     ########
#####################################################################

# getting nicknamesArray from csv
with open("data/BASICDATA1.csv") as nicknamesFile:
    reader = csv.DictReader(nicknamesFile, delimiter=',')
    for line in reader:
        nicknamesArray.append(line["usernames"])
nicknamesArray = pd.read_csv('nicknames.csv').iloc[:,0].tolist()
print('Starting parsing tweets...')
#getting followers tweets
d = {}
df_data = {}
df_data["NICKNAME"] = []
df_data["LANG"] = []
df_data["LOT"] = []
df_data["TWEETS"] = []
tweetsPerUser = 6

# nicknamesArray = nicknamesArray[:5]
for j in range(len(nicknamesArray)):
    tweetsarr = []
    df_tweersarr = []
    nickName = nicknamesArray[j]
    followersUrl = "https://twitter.com/"+nickName

    driver.get(followersUrl)
    print('----------------++++++-------------------')
    print('Starting parsing ' +nickName +'`s tweets!')
    print('It is ' +str(j) + ' user of ' + str(len(nicknamesArray)) )

    last_height = driver.execute_script("return document.body.scrollHeight")
    print('Iter')
    for i in range(1,tweetsPerUser):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        new_height = driver.execute_script("return document.body.scrollHeight")
        # if new_height == last_height: ####!!!!!! refactor this place
        #     break
        last_height = new_height

        print(str(i), end=" ", flush=True)
        time.sleep(3)

        html_source = driver.page_source
        sourcedata= html_source.encode('utf-8')
        soup=bs(sourcedata, features="html.parser")
        elements = [x.text.strip().replace('\n', ' ') for x in soup.body.findAll('div', 'css-901oao r-18jsvk2 r-1qd0xha r-a023e6 r-16dba41 r-rjixqe r-bcqeeo r-bnwqim r-qvutc0')]
        if len(elements) == 0:
            print('No tweets was found')
            break
        tweetsarr.extend(list(elements))
        # print('Total '+ str(len(tweetsarr)))

    time.sleep(3)

    d[nickName] = tweetsarr
    if (len(tweetsarr) > 0):
        df_data["NICKNAME"].append(nickName)
        df_data["TWEETS"].append(' ||| '.join(tweetsarr))
        df_data["LANG"].append('23.232323')
        df_data["LOT"].append('34.343434')
    # df = pd.DataFrame(data=d)
    # df.to_csv('data/'+ nickName +'.csv', mode='a', header=False)
    time.sleep(2)
print('----------------------------------------------------')
print('End of a program!')

# with open('AllTweetsCsv.csv', 'w', newline='', encoding='utf-8') as file:
# 	writer = csv.DictWriter(file, fieldnames=d)
# 	writer.writeheader()
# 	writer.writerow(d)

with open("AllTweetsJson.json", "w", encoding="utf-8") as file:
    json.dump(d, file)
print('-----------------------------------------------------')
# print(df_data)
df = pd.DataFrame(data=df_data)
df.to_csv('AlltweeetsBigFile.csv', sep='	', index=False, header=False)

compression_opts = {'method':'gzip',
                        'archive_name':'user_info.test.csv'}
df.to_csv('user_info.test.gz', sep='	', index=False, header=False,
          compression='gzip')
