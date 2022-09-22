
import requests
import lxml
import pandas as pd
from bs4 import BeautifulSoup as bs
from selenium import webdriver
import time

# dates=[d.strftime('%Y-%m-%d') for d in pd.date_range('20220901','20220920')]
dates=['2022-09-21']

try:
    nf_movie_df=pd.read_csv('/Users/ihyeon-u/Desktop/Netflex_Movie_data.csv',index_col=0)
    movie_existence=True
except:
    movie_existence=False
    
try:
    nf_tvshow_df=pd.read_csv('/Users/ihyeon-u/Desktop/Netflex_TV_Show_data.csv',index_col=0)
    tvshow_existence=True
except:
    tvshow_existence=False

print('movie:',movie_existence,'tvshow:',tvshow_existence)
driver=webdriver.Chrome('./chromedriver')  #크롬 드라이버를 하지 않으려면 webdriver-manager 설치
driver.implicitly_wait(3)

for date in dates:
    
    print(date)
    

    

    url="https://flixpatrol.com/top10/netflix/world/"+str(date)+"/full/#netflix-1"
    driver.get(url)
    time.sleep(1)

    html=driver.page_source
    soup=bs(html,'lxml')
    movie_chart=soup.select('#netflix-1 > div.-mx-content > div > div > table > tbody > tr')
    tv_show_chart=soup.select('#netflix-2 > div.-mx-content > div > div > table > tbody > tr')
    movie_data=dict()
    tv_show_data=dict()

    for i in range(len(movie_chart)):
        movie_name=movie_chart[i].find('a')
        movie_data[i+1]=movie_name.text.strip()

    for i in range(len(tv_show_chart)):
        tv_show_name=tv_show_chart[i].find('a')
        tv_show_data[i+1]=tv_show_name.text.strip()

    if movie_existence:
        movie_df=pd.DataFrame(pd.Series(movie_data),
                                columns=[date],dtype=object)

        nf_movie_df=pd.concat([nf_movie_df,movie_df],axis=1)

    else:
        nf_movie_df=pd.DataFrame(movie_data.values(), index=movie_data.keys(), columns=[date])
        movie_existence=True

    if tvshow_existence:
        tvshow_df=pd.DataFrame(pd.Series(tv_show_data),
                                columns=[date],dtype=object)
        nf_tvshow_df=pd.concat([nf_tvshow_df,tvshow_df],axis=1)

    else:
        nf_tvshow_df=pd.DataFrame(tv_show_data.values(), index=tv_show_data.keys(), columns=[date])
        tvshow_existence=True

nf_movie_df.to_csv('Netflex_Movie_data.csv')
nf_tvshow_df.to_csv('Netflex_TV_Show_data.csv')

#netflix-1 > div.-mx-content > div > div > table > tbody > tr:nth-child(1) > td:nth-child(3) > a

