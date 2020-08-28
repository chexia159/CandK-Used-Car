# ==================================
# @Time        :Jun 28 2020
# @Author      :Da Ban
# @FileName    :168_UC.py
# @Software    :Visual Studio Code
# ==================================

# import libraries
from bs4 import BeautifulSoup
import requests
from lxml import etree
import csv
import random
import time
import re
import sys

# parse detailed URL for each car
def get_detail_url(url):
    # query the website and return the html to variable 'reponse'
    response = requests.get(url)
    ##print(response.text)
    # parse the html using beautiful soup and store in variable 'soup'
    soup = BeautifulSoup(response.text,'html.parser')
    ## MSRP_infos = soup.select('a.carinfo') # handle it later
    ##MSRP = MSRP_infos[0].select_one('s').text
    ##MSRP = re.findall(r'\d+\.\d+',MSRP)
    ##print(soup)
    # find the car list
    carlist = soup.find('ul', attrs={'class': 'viewlist_ul'})
    results = carlist.find_all('li')
    ##print('Number of the results', len(results))
    # get detailed URL for each car
    detail_urls = []
    for result in results:
        # find the URL
        detail_url = result.find('a').get('href')
        detail_urls.append(detail_url)
        ##print(len(detail_urls))
    if len(detail_urls) == 0:
        print("Error! Didn't find detail urls")
        sys.exit()
    return detail_urls

# parse detail info for each car
def parse_detail_page(detail_url):
    # create a base url
    base_url = 'https://www.che168.com'
    # query the website and return the html to variable 'detail_reponse'
    detailurl = base_url + detail_url
    detail_response = requests.get(url=detailurl)
    # parse
    detail_soup = BeautifulSoup(detail_response.text,'html.parser')
    name_info = detail_soup.find('div', attrs={'class': 'car-box'})
    if name_info != None:
        car_name = name_info.find('h3', attrs = {'class':'car-brand-name'})
        name = car_name.getText()
        ##name = name.partition(' ')[0]
        ##name = name.strip('\n')
        name_info = None
        car_info = detail_soup.find('ul', attrs={'class': 'brand-unit-item fn-clear'})
        # find the info columns
        car_data = car_info.find_all('h4') # an array with mileage, registertime, Transmission/Displacement, location, inspection
        TandD = car_data[2].getText()
        transmission = TandD.split('/')[0]
        transmission = transmission.strip()
        displacement = TandD.split('/')[1]
        displacement = displacement.strip() 
        car_price_info = detail_soup.find('div', attrs={'class': 'brand-price-item'})
        car_price = car_price_info.getText()
        ##print(car_price)
        # car_price
        car_price = re.findall(r'\d+\.\d+',car_price)
        car_price = ''.join(car_price)
        ##print(car_price)
        basic_info = detail_soup.find('div', attrs={'class': 'all-basic-content fn-clear'})
        basic_data = basic_info.find_all('li')
        # trim all the necessary
        color = basic_data[14].getText()
        color = color.split('颜色')[1]
        record = basic_data[9].getText()
        warranty = basic_data[8].getText()
        inspection = car_data[4].getText()
        inspection = inspection.split('\r')[0]
        # translate to Eng
        ##if name == "领动":
        ##    name = "LD"
        ##if color == "白色":
        ##    color = 1
        ##else:
        ##    color = 0
        ##if transmission == "自动":
        ##    transmission = "AT"
        mil = car_data[0].getText()
        mileage = re.findall(r'\d+\.\d+',mil)
        if len(mileage) == 0:
            mileage = re.findall(r'\d+',mil)
        mileage = ''.join(mileage)
        RT = car_data[1].getText()
        RT = RT.encode('ascii',errors = 'ignore')
        RT = RT.decode()
        RT = RT[:4]+'-'+RT[4:]
        # package
        infos = {}
        infos['Model'] = name
        infos['Price'] = car_price
        ##infos['MSRP']
        infos['Color'] = color
        infos['Transmission'] = transmission
        infos['Displacement'] = displacement
        infos['Mileage'] = mileage
        infos['Register Time'] = RT
        infos['Location'] = car_data[3].getText()
        infos['Inspection'] = inspection
        infos['Warranty'] = warranty
        infos['Record'] = record 
        return infos
    else:
        error = 1
        return error

# save data
def save_data(data,f):
    f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(data['Model'],data['Price'],
    data['Color'],data['Transmission'],data['Displacement'],
    data['Mileage'],data['Register Time'],data['Location'],data['Inspection'],data['Warranty'],data['Record'])) # MSRP

# loop over results
def main():
    # https://www.che168.com/china/xiandai/lingdong/a0_0msdgscncgpi1ltocsp1exx0/?pvareaid=102179#currengpostion
    with open('168_Used_Car_LD.csv','a',encoding='utf-8-sig', errors = 'ignore') as f:
        for i in range(1,14):
            num = 0
            print('正在爬取第' + str(i) + '页数据...')
            url = 'https://www.che168.com/china/xiandai/lingdong/a0_0msdgscncgpi1ltocsp' + str(i) +'exx0/?pvareaid=102179#currengpostion'
            detail_urls = get_detail_url(url)
            for detail_url in detail_urls:
                if detail_url.startswith('https'):
                    continue
                data = parse_detail_page(detail_url)
                num += 1
                try:
                    save_data(data,f)
                     ## num += 1
                    print('第' + str(num) + '条数据爬取完毕')
                     # pause
                    time.sleep(1)
                except:
                    data = 1
            print('第' + str(i) + '页数据爬取完毕')
            print('=============================')
            time.sleep(random.randint(1,5))
        print('Mission Complete')
# if the main
if __name__ == '__main__':
    main()