# ==================================
# @Time        :Aug 3 2020
# @Author      :Da Ban
# @FileName    :K_Test_Ajax.py
# @Software    :Visual Studio Code
# ==================================

# import libraries
import requests
from urllib.parse import urlencode
from pyquery import PyQuery as pq
import csv
import time

# specify the base url
base_url = 'https://api.encar.com/search/car/list/premium?'
headers = {
    'authority':'api.encar.com',
    'origin':'http://www.encar.com',
    'referer':'http://www.encar.com/dc/dc_carsearchlist.do?carType=kor',
    'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36',
    }

# specify the query parameters
def get_sr(sr):
    params = {
        'count':'true',
        'q':'(And.Hidden.N._.(C.CarType.Y._.(C.Manufacturer.현대._.(C.ModelGroup.아반떼._.(C.Model.아반떼 AD._.BadgeGroup.가솔린 1600cc.)))))',
        'sr': sr
    }
    url = base_url + urlencode(params)
    try: 
        response = requests.get(url, headers = headers)
        if response.status_code == 200:
            return response.json(),sr
    except requests.ConnectionError as e:
        print('Error', e.args)

# parse details
def parse_page(json, sr:str):
    if json:
        items = json.get('SearchResults')
        if items != None:
            num = 1
            for item in items:
                infos = {}
                infos['trim'] = item.get('Badge')
                infos['condition'] = item.get('Condition')
                infos['fueltype'] = item.get('FuelType')
                infos['mileage'] = item.get('Mileage')
                infos['model'] = item.get('Model')
                infos['price'] = item.get('Price')
                infos['transmission'] = item.get('Transmission')
                infos['trust'] = item.get('Trust')
                infos['year'] = item.get('Year')
                ##print(infos)
                save_data(infos,f)
                print(str(num)+'data saved')
                num += 1
            return infos
        else:
            error = 1
            return error

# save data
def save_data(result,f):
    f.write('{},{},{},{},{},{},{},{},{}\n'.format(result['trim'],result['condition'],
    result['fueltype'],result['mileage'],result['model'],
    result['price'],result['transmission'],result['trust'],result['year']))

if __name__ == '__main__':
    with open('K_Used_Car.csv','a',encoding='utf-8-sig', errors = 'ignore') as f:
        for i in range(0,20):

            count = i * 50
            sr = '|ModifiedDate|'+ str(count) +'|50'
            json = get_sr(sr)
            results = parse_page(*json)
            time.sleep(10)
        print('Mission Completed')

            
