from bs4 import BeautifulSoup, Tag
from selenium import webdriver


driver1 = webdriver.Chrome("E:/Tools/chromeDriver/chromedriver.exe")
driver2 = webdriver.Chrome("E:/Tools/chromeDriver/chromedriver.exe")
main_url = "https://elaw.klri.re.kr"

initial_hseq = 1

for hseq in range(initial_hseq, 50000):
    url = main_url + "/kor_service/lawView.do?hseq=%s&lang=KOR" % hseq
    driver1.get(url)

    # 페이지 소스 얻어서 구조화 시키기
    source = driver1.page_source
    soup = BeautifulSoup(source, "html.parser")

    # 현재 url에서 접근 가능한 iframe까지 접근. 여기서 src값 얻기.
    iframe_str = str(soup.find("iframe", {"id": "lawViewContent"}))
    iframe_src_url = iframe_str.split('src="')[1].split('"')[0]

    # 실제 사용할 url 코드 얻기
    wanted_url = main_url + iframe_src_url
    driver2.get(wanted_url)
    source2 = driver2.page_source
    soup2 = BeautifulSoup(source2, "html.parser")

    # 원하는 영역으로 접근
    wanted_area = soup2.find("div", {"class": "lawmultiview"})

    # 가끔 lawmultiview라는 div가 없음
    if not wanted_area:
        continue

    tr_list = wanted_area.contents[1].contents[3].contents

    # tr_list가 4개라는 것은 내용이 없다는 뜻
    if len(tr_list) == 4:
        continue

    for tr in tr_list:
        if type(tr) != Tag:
            continue

        td_eng = tr.contents[1]
        td_kor = tr.contents[3]

        text_eng = td_eng.text.strip()
        text_kor = td_kor.text.strip()

        if not text_eng or not text_kor:
            continue

        print("=" * 50)
        print(text_eng.replace("\n", "\t"))
        print(text_kor.replace("\n", "\t"))

    print("a")

