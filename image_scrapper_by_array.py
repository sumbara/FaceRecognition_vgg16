from bs4 import BeautifulSoup
import urllib.request
from selenium import webdriver
import random, time, os


# 크롬 사용 시 옵션 들을 설정
options = webdriver.ChromeOptions()
# options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument('disable-gpu')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.87 Safari/537.36")
# 크롤링 할 브라우저로 크롬을 사용할 것을 설정
driver = webdriver.Chrome('F:\PycharmProjects\opencv_test\chromedriver', chrome_options=options)

names = ['Dominik Livakovic', 'Duje Caleta-Car', 'Filip Bradaric', 'Josip Pivaric', 'Lovre Kalinic', 'Milan Badelj', 'Nikola Kalinic']

for name in names:
    keyword = name
    i = 1
    if keyword == 'exit':
        break
    if ' ' in keyword:
        url_keyword = keyword.split(' ')
        url_keyword = "%20".join(url_keyword)
    else :
        url_keyword = keyword
    # 검색할 URL 지정
    url = 'https://www.bing.com/images/search?q=' + url_keyword + '&FORM=HDRSC2'
    header = {'User-Agent': "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.87 Safari/537.36"}
    down_cnt = 0
    try:
        print(url)
        driver.get(url)
        for i in range(6):
            # URL 이미지 페이지의 스크롤을 변경. 검색된 URL에서 이미지를 다 다운받고 스크롤을 변경해서 더 다운받기 위해
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
        # BeautifulSoup를 이용해서 해당 페이지의 소스를 XML 파일로 읽어들인다.
        page_soup = BeautifulSoup(driver.page_source, 'lxml')

        # 읽어들인 XML 파일로부터 해당 태그(클래스 포함)가 적용된 녀석들을 찾는다.
        image_links = page_soup.find_all('a', attrs={'class': 'iusc'})
        print('찾은 이미지 링크 개수 : ' + str(image_links.__len__()))

        for image_link in image_links:
            image_link_href = image_link.get('href')
            if image_link_href is not None:
                req_page = urllib.request.Request('https://www.bing.com' + image_link_href, headers=header)
                html = urllib.request.urlopen(req_page)
                page_soup = BeautifulSoup(html, 'lxml')
                images = page_soup.find_all('img')

                for real_image in images:
                    if real_image.get('data-reactid') is not None:
                        if '&w=60&h=60&c=7' not in real_image.get('src'):
                            real_image_src = real_image.get('src')
                            if not os.path.isdir("F:\SoccerPlayer_image_2\Croatia/" + keyword + "/"):
                                os.mkdir("F:\SoccerPlayer_image_2\Croatia/" + keyword + "/")
                            full_name = "F:\SoccerPlayer_image_2\Croatia/" + keyword + "/" + str(i) + ".jpg"
                            i += 1
                            down_cnt += 1

                        urllib.request.urlretrieve(real_image_src, full_name)
            else:
                pass
    except urllib.request.HTTPError:
        print('error')
    print('다운받은 이미지 수 : ' + str(down_cnt))