from bs4 import BeautifulSoup, Tag
from selenium import webdriver


driver1 = webdriver.Chrome("E:/Tools/chromeDriver/chromedriver.exe")
driver2 = webdriver.Chrome("E:/Tools/chromeDriver/chromedriver.exe")
main_url = "https://elaw.klri.re.kr"

out_list = [] # 결과 저장 리스트
previous_law_title = "" # 직전에 성공한 법의 제목
initial_hseq = 1

for hseq in range(initial_hseq, 50000):
    print("~~~~~ hseq = %s" % hseq)

    url = main_url + "/kor_service/lawView.do?hseq=%s&lang=KOR" % hseq
    driver1.get(url)

    # 페이지 소스 얻어서 구조화 시키기
    source = driver1.page_source
    soup = BeautifulSoup(source, "html.parser")

    # 현재 url에서 접근 가능한 iframe까지 접근. 여기서 src값 얻기.
    try:
        iframe_str = str(soup.find("iframe", {"id": "lawViewContent"}))
        iframe_src_url = iframe_str.split('src="')[1].split('"')[0]
    except Exception as e:
        continue

    # 실제 사용할 url 코드 얻기
    wanted_url = main_url + iframe_src_url
    driver2.get(wanted_url)
    source2 = driver2.page_source
    soup2 = BeautifulSoup(source2, "html.parser")

    # 원하는 영역으로 접근
    wanted_area = soup2.find("div", {"class": "lawmultiview"})

    # 가끔 lawmultiview라는 div가 없음. 이럴 때는 그냥 continue
    if not wanted_area:
        continue

    # 법 제목으로 이미 크롤링 성공했던 것은 다시 하지 않기
    current_law_title = wanted_area.contents[1].contents[3].text.strip().split("\n")[0]

    if previous_law_title == current_law_title:
        continue

    tr_list = wanted_area.contents[1].contents[3].contents

    # tr_list가 4개라는 것은 내용이 없다는 뜻
    if len(tr_list) == 4:
        continue

    for tr in tr_list:
        if type(tr) != Tag:
            continue

        # 조 단위로 묶인 내용
        try:
            td_eng = tr.contents[1]
            td_kor = tr.contents[3]
        except IndexError as e:
            continue

        # 내용 없으면 continue
        if not td_eng.text.strip() or not td_kor.text.strip():
            continue

        # 조제목 텍스트 얻기
        article_title_eng = td_eng.find("div", {"class": "articletitle"})
        article_title_kor = td_kor.find("div", {"class": "articletitle"})

        # 조제목이 없으면 그냥 text 바로 출력 후 continue
        if not article_title_eng or not article_title_kor:
            # 아무래도 이상한 게 많아서 그냥 조항쪽만 출력하기로.
            # print("=" * 50)
            # print(td_eng.text.strip())
            # print(td_kor.text.strip())

            continue

        # 조항 얻기
        hang_eng = td_eng.find_all("div", {"class": "hang"})
        hang_kor = td_kor.find_all("div", {"class": "hang"})

        if not hang_eng or not hang_kor:
            hang_eng = td_eng.find_all("div", {"class": "none"})
            hang_kor = td_kor.find_all("div", {"class": "none"})

        hang_eng_text_list = []
        hang_kor_text_list = []

        for i in range(min(len(hang_eng), len(hang_kor))):
            hang_eng_text = hang_eng[i].text
            hang_kor_text = hang_kor[i].text

            # # 조항의 숫자 빼고 내용만 얻기
            # for j in range(len(hang_eng[i])):
            #     hang_cont_eng = hang_eng[i].contents[j].find("span")
            #
            #     if hang_cont_eng:
            #         hang_eng_text += hang_cont_eng.text
            #     else:
            #         hang_eng_text += hang_eng[i].contents[j].text
            #
            # for j in range(len(hang_kor[i])):
            #     hang_cont_kor = hang_kor[i].contents[j].find("span")
            #
            #     if hang_cont_kor:
            #         hang_kor_text += hang_cont_kor.text
            #     else:
            #         hang_kor_text += hang_kor[i].contents[j].text

            hang_eng_text_list.append(hang_eng_text.strip())
            hang_kor_text_list.append(hang_kor_text.strip())

        # 출력
        # print("=" * 50)
        # print(article_title_eng.text.strip())
        # print(article_title_kor.text.strip())
        # print("#" * 30)

        for i in range(len(hang_eng_text_list)):
            # print(hang_eng_text_list[i])
            # print(hang_kor_text_list[i])

            with open("datasource/raw_text.txt", "a") as f:
                f.write(hang_kor_text_list[i].replace("\t", " ") + "\t" + hang_eng_text_list[i].replace("\t", " ") + "\n")

    previous_law_title = current_law_title

    # print("a")

