# -*- coding: utf-8 -*-
import os
import base64
from datetime import datetime
import cv2
import json
import sys
import operator
import re
import math
import http.client, urllib.request, urllib.parse, urllib.error, base64
from flask import Flask, render_template, request, send_file
from werkzeug import secure_filename
from difflib import SequenceMatcher
from pdf2image import convert_from_path, convert_from_bytes
sys.path.append('C:/projectWork/icrRest/labelTrain')
sys.path.append('C:/projectWork/icrRest/entryTrain')
import labelTrain
import labelEval
import entryTrain
import entryEval

app = Flask(__name__)

labelFileName = '/home/taihoinst/icrRest/labelTrain/data/kkk.train'
labelFileList = '/home/taihoinst/icrRest/labelTrain/data/kkk.cls'

entryFileName = '/home/taihoinst/icrRest/entryTrain/data/kkk.train'
entryFileList = '/home/taihoinst/icrRest/entryTrain/data/kkk.cls'

regExp = "[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]"

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/upload")
def render_file():
    return render_template('upload.html')

@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        upload_path = 'C:/ICR/uploads/'

        retResult = []
        obj = {}
        f = request.files['file']
        ext = os.path.splitext(f.filename)[1]
        convertFilename = "tempFileName_" + datetime.today().strftime("%Y%m%d%H%M%S") + ext
        #f.save(secure_filename(convertFilename))
        f.save(os.path.join(upload_path, convertFilename))

        if ext == ".pdf":
            fileNames = convertPdfToImage(upload_path, convertFilename)
            print(fileNames)
            for item in fileNames:
                imgResize(upload_path + item)
                obj = pyOcr(upload_path + item)
                retResult.append(obj)
        else:
            fileNames = imgResize(upload_path + convertFilename)
            for item in fileNames:
                obj = pyOcr(item)
            retResult.append(obj)

        result = re.sub('None', "null", json.dumps(retResult, ensure_ascii=False))
        return str(result)
    else:
        return "upload GET"

@app.route("/fileDown", methods = ['GET', 'POST'])
def download_file():
    upload_path = '/home/taihoinst/icrRest/uploads/'
    fileName = request.args.get('fileName')
    
    with open((upload_path + fileName), 'rb') as single_img:
        img_b64 = base64.b64encode(single_img.read())
    return img_b64

@app.route("/insertDocSentence", methods = ['GET','POST'])
def insertDocSentence():
    if request.method == 'POST':
        str = request.form['sentence']
        print(str)
        file = open('docSentence.txt', 'a', -1, encoding='UTF8')
        file.write('\n')
        file.write(str)
        file.close()

        return "success"
    else:
        return render_template('insertDocSentence.html')

@app.route("/insertSplitData", methods = ['GET','POST'])
def insertSplitData():
    if request.method == 'POST':
        data = request.get_json()
        sentence = data['sentence']
        sentence = json.loads(sentence)

        file = open('splitLabel.txt', 'a', -1, encoding='UTF8')
        for item in sentence:
            file.write("\n" + item)
        file.close()

        return "success"

@app.route("/insertEntry", methods = ['GET','POST'])
def insertEntry():
    if request.method == 'POST':
        data = request.get_json()
        entry_rslt = filteredEntryData(data)
        """
        sentence = data['sentence']
        sentence = json.loads(sentence)
        file = open('splitLabel.txt', 'a', -1, encoding='UTF8')
        for item in sentence:
            file.write("\n" + item)
        file.close()
        """
        if entry_rslt:
            entryTrain.startTrain()

        return "success"

@app.route("/insertLabelCol", methods = ['GET','POST'])
def insertLabelCol():
    print(":::method IN:::")
    if request.method == 'POST':
        print("method POST:::")
        data = request.get_json()
        print("data:::", data)
        finalArr = filteredData(data)
        return "success"

    else:
        return "fail"

@app.route('/labelTrain', methods = ['POST'])
def labelTrainEx():
    if request.method == 'POST':
        labelTrain.startTrain()
        return "labelTrain"

@app.route('/entryTrain', methods = ['POST'])
def entryTrainEx():
    if request.method == 'POST':
        entryTrain.startTrain()
        return "entryTrain"

@app.route('/labelEval', methods = ['POST'])
def labelEvalEx():
    if request.method == 'POST':
        ocrData = request.form['ocrData']
        print(ocrData)
        ocrData = json.loads(ocrData)
        ocrData = labelEval.startEval(ocrData)
        return str(ocrData)

@app.route('/entryEval', methods = ['POST'])
def entryEvalEx():
    if request.method == 'POST':
        ocrData = request.form['ocrData']
        ocrData = json.loads(ocrData)
        ocrData = entryEval.startEval(ocrData)
        return str(ocrData)

def filteredEntryData(apiData):
    try:
        #dyyoo
        data = apiData
        sentence = data['sentence']
        #print("data['value']:::", data['value'])
        sentence = json.loads(sentence)
        # print(":::")
        print("sentence:::", sentence)

        readFile = readJson(entryFileName)
        if readFile:
            print("readFile:::Success")
        else:
            print("readFile:::Empty")
            readFile = []

        print(":::")
        #print("sentence:::", readFile)

        new_entry_list = []
        new_entry_col_list = []
        for sObj in sentence:
            is_written = True
            is_new = True
            print("1sObj:::", sObj)
            split_entry_str = sObj.split(",")
            split_entry_str0 = split_entry_str[0].replace('\\n', '').strip()
            split_entry_str1 = split_entry_str[1].replace('\\n', '').strip()
            new_text = split_entry_str0 + ',' + split_entry_str1

            for jObj in readFile:
                split_str = jObj.split(",")
                split_str0 = split_str[0].replace('\\n', '').strip()
                split_str1 = split_str[1].replace('\\n', '').strip()

                if split_str1 == split_entry_str1:
                    is_new = False
                    if split_str0 == split_entry_str0:
                        is_written = False

            if is_written:
                new_entry_list.append(new_text)
                writeJson(new_text, entryFileName)
            if is_new:
                is_new_col = True
                for newCol in new_entry_col_list:
                    if newCol == split_entry_str1:
                        is_new_col = False
                        break
                if is_new_col:
                    print("new col-col_lbl:::", split_entry_str1)
                    writeJson(split_entry_str1, entryFileList)
                    #new_entry_col_list.append(split_entry_str1)

        #print("new_entry_col_list:::", new_entry_col_list)
        #print("new_entry_list:::", new_entry_list)
        return new_entry_list
    except Exception as e:
        print(e)
        return []

def filteredData(apiData):
    try:
        #dyyoo
        data = apiData
        sentence = data['value']
        #print("data['value']:::", data['value'])
        sentence = json.loads(sentence)
        # print(":::")
        # print("sentence:::", sentence)

        readFile = readJson(labelFileName)
        if readFile:
            print("readFile:::Success")
        else:
            print("readFile:::Empty")
            readFile = []

        print(":::")
        #print("sentence:::", readFile)

        new_col_list = []
        for sObj in sentence:
            is_written = True
            is_new = True
            col_lbl = sObj['colLbl'].replace('\\n', '').strip()
            col_text = sObj['text'].replace('\\n', '').strip()

            for jObj in readFile:
                split_str = jObj.split(",")
                split_str0 = split_str[0].replace('\\n', '').strip()
                split_str1 = split_str[1].replace('\\n', '').strip()

                if split_str0 == col_text and split_str1 == col_lbl:
                    is_written = False
                if split_str1 == col_lbl:
                    is_new = False

            # print("is_written:::", is_written)
            # print("is_new:::", is_new)
            if is_written:
                print("col_text:::", col_text)
                print("col_lbl:::", col_lbl)
                new_text = col_text + ',' + col_lbl
                writeJson(new_text, labelFileName)
            if is_new:
                is_new_col = True
                for newCol in new_col_list:
                    if newCol == col_lbl:
                        is_new_col = False
                        break
                if is_new_col:
                    print("new col-col_lbl:::", col_lbl)
                    new_col_list.append(col_lbl)

        print("new_col_list:::", new_col_list)
        if new_col_list:
            for newCol in new_col_list:
                writeJson(newCol, labelFileList)
            # label ì¶ê? ???ìµ
            labelTrain.startTrain()
        return readFile
    except Exception as e:
        print(e)
        return {}

def readJson(readLoc):
    try:
        print("readJson:::IN")
        #with open(readLoc, 'rU', encoding="utf-16") as data:
        with open(readLoc, 'rb') as data:
            lines = [l.decode('utf8', 'ignore') for l in data.readlines()]
        print(lines)
        return lines
    except Exception as e:
        print(e)
        return []

def writeJson(f_data, file_path):
    try:
        file = open(file_path, 'a', -1, encoding='UTF8')
        file.write('\n')
        file.write(f_data)
        file.close()
        #return True
    except Exception as e:
        print(e)
        #eturn False


def pyOcr(item):
    # item = 'C:/ICR/uploads/tempFileName_20190507141212-0.jpg'
    item = 'C:/ICR/uploads/daelimTest.jpg'
    # MS ocr api 호출
    # ocrData = get_Ocr_Info(item)
    ocrData = json.loads('[{"location": "267,377,199,66", "text": "1-174"}, {"location": "643,453,591,66", "text": "레디 리스트 콘크리트"}, {"location": "1070,2135,197,36", "text": "산 골 재 율"}, {"location": "280,649,557,34", "text": "0표 준 명 : 레디믹스트 콘크리트"}, {"location": "280,690,410,32", "text": "0표 준 번 호 . KS F 4009"}, {"location": "280,731,480,32", "text": "0인 증 번 호 . 제96-03-026호"}, {"location": "280,772,431,33", "text": "0인 증 기 관 : 한국표준협회"}, {"location": "280,813,643,35", "text": "0인| 증 종 류 : 보통,포장,고강도 콘크리트"}, {"location": "350,883,496,42", "text": "20 18년 11 월 08 일"}, {"location": "880,940,74,41", "text": "귀하"}, {"location": "1090,667,144,36", "text": "등록번호"}, {"location": "1027,720,207,48", "text": "亐 상 호"}, {"location": "1026,810,205,39", "text": "급 성 명"}, {"location": "1199,890,34,30", "text": "소"}, {"location": "1026,916,35,36", "text": "자"}, {"location": "1090,965,146,35", "text": "대표전화"}, {"location": "288,941,287,48", "text": "대림산업(주)"}, {"location": "403,1051,74,40", "text": "013"}, {"location": "289,1065,36,25", "text": "No"}, {"location": "1853,269,84,54", "text": "曇."}, {"location": "1274,455,176,64", "text": "납품서"}, {"location": "1312,554,163,63", "text": "4體."}, {"location": "1342,652,379,32", "text": "132-81-13908"}, {"location": "1341,722,364,51", "text": "주식회사 é!0"}, {"location": "1683,729,179,75", "text": "•貳&譬4"}, {"location": "1274,803,406,44", "text": "대표이사 전 찬 7"}, {"location": "1273,885,475,39", "text": "경기도 남양주시 와부옵 수레로"}, {"location": "1262,968,557,36", "text": "031-576-4545 출하실 031-576-3131"}, {"location": "1142,1052,528,71", "text": "타설완료• 0응 시 수"}, {"location": "287,1153,694,44", "text": "납 품 장 소 고덕 대림 아파트 현장"}, {"location": "286,1252,194,37", "text": "운반차번호"}, {"location": "287,1392,194,36", "text": "납 품 시 각"}, {"location": "663,1248,404,44", "text": "(175) 서울14다7478"}, {"location": "858,1342,46,36", "text": "08"}, {"location": "1439,1252,129,39", "text": "이은우"}, {"location": "1247,1342,45,36", "text": "03"}, {"location": "543,1439,133,36", "text": "도 착"}, {"location": "287,1525,191,37", "text": "납 품 용 적"}, {"location": "751,1522,91,37", "text": "6.00"}, {"location": "1185,1529,31,35", "text": "계"}, {"location": "559,1600,423,33", "text": "콘크리트의 굵은골재의 최대"}, {"location": "1056,1623,129,32", "text": "호칭강도"}, {"location": "536,1644,449,33", "text": "종류에 따른 구분 치수에 따른 구분"}, {"location": "284,1662,193,37", "text": "호 칭 방 법"}, {"location": "535,1711,211,31", "text": "보통콘크리트"}, {"location": "854,1714,374,37", "text": "25 mm 18 ,••曲"}, {"location": "877,1799,323,41", "text": "시방 배합표(kg/m3)"}, {"location": "1277,1603,171,31", "text": "슬럼프 또는"}, {"location": "1276,1646,172,31", "text": "슬럼프 플로"}, {"location": "1547,1524,91,36", "text": "6.00"}, {"location": "1762,1531,34,35", "text": "m3"}, {"location": "1552,1604,199,32", "text": "시멘트 종류에"}, {"location": "1585,1648,137,31", "text": "따른 구분"}, {"location": "1496,1697,309,31", "text": "포를랜드시멘트 1종"}, {"location": "1324,1714,146,37", "text": "150 mm"}, {"location": "258,1877,181,28", "text": "시멘트 시멘트"}, {"location": "480,1887,26,26", "text": "물"}, {"location": "551,1878,1267,36", "text": "회수수 잔골재 잔골재 산골재 굵은골재굵은골재 굵은골재 혼화재 혼화재 혼화재 혼화제 혼화제 혼화제"}, {"location": "683,1916,322,26", "text": "0 ㉣ ㉭ 0"}, {"location": "1178,1918,23,24", "text": "㉭"}, {"location": "1278,1917,511,25", "text": "0 ㉣ ㉭ ㉭"}, {"location": "691,2017,149,25", "text": "467 467"}, {"location": "742,2135,82,32", "text": "59.6"}, {"location": "544,2314,418,42", "text": "염화물량 : 0.3kg/m3 이하"}, {"location": "992,2017,28,25", "text": "91"}, {"location": "1302,2017,31,25", "text": "43"}, {"location": "1569,2017,46,25", "text": "2.0"}, {"location": "1482,2135,82,33", "text": "51.2"}, {"location": "269,2133,223,36", "text": "물결합재비"}, {"location": "269,2225,227,37", "text": "지 정 사 항"}, {"location": "460,2320,33,26", "text": "고"}, {"location": "274,2407,213,36", "text": "인수자 확인"}, {"location": "462,2498,31,36", "text": "타"}, {"location": "249,2648,268,28", "text": "B5(182mm ※257mm)"}, {"location": "1790,2140,26,27", "text": "0/0"}, {"location": "1340,2218,121,67", "text": "7%한"}, {"location": "1453,2311,260,40", "text": "4.5 士 15%"}, {"location": "1233,2317,124,37", "text": "공기랑•"}, {"location": "1067,2386,198,36", "text": "출하계 확인"}, {"location": "1442,2406,120,36", "text": "인 성훈"}, {"location": "1058,2430,216,36", "text": "표시사항확인"}, {"location": "540,2490,993,42", "text": "(성유보강제)상일동역사거리 직진 3-7게이트 329동"}, {"location": "967,2616,289,47", "text": "주식회사 산하"}]')
    # Y축정렬
    # ocrData = sortArrLocation(ocrData)
    ocrData = json.loads('[{"location": "1853,269,84,54", "text": "曇."}, {"location": "267,377,199,66", "text": "1-174"}, {"location": "643,453,591,66", "text": "레디 리스트 콘크리트"}, {"location": "1274,455,176,64", "text": "납품서"}, {"location": "1312,554,163,63", "text": "4體."}, {"location": "280,649,557,34", "text": "0표 준 명 : 레디믹스트 콘크리트"}, {"location": "1342,652,379,32", "text": "132-81-13908"}, {"location": "1090,667,144,36", "text": "등록번호"}, {"location": "280,690,410,32", "text": "0표 준 번 호 . KS F 4009"}, {"location": "1027,720,207,48", "text": "亐 상 호"}, {"location": "1341,722,364,51", "text": "주식회사 é!0"}, {"location": "1683,729,179,75", "text": "•貳&譬4"}, {"location": "280,731,480,32", "text": "0인 증 번 호 . 제96-03-026호"}, {"location": "280,772,431,33", "text": "0인 증 기 관 : 한국표준협회"}, {"location": "1274,803,406,44", "text": "대표이사 전 찬 7"}, {"location": "1026,810,205,39", "text": "급 성 명"}, {"location": "280,813,643,35", "text": "0인| 증 종 류 : 보통,포장,고강도 콘크리트"}, {"location": "350,883,496,42", "text": "20 18년 11 월 08 일"}, {"location": "1273,885,475,39", "text": "경기도 남양주시 와부옵 수레로"}, {"location": "1199,890,34,30", "text": "소"}, {"location": "1026,916,35,36", "text": "자"}, {"location": "880,940,74,41", "text": "귀하"}, {"location": "288,941,287,48", "text": "대림산업(주)"}, {"location": "1090,965,146,35", "text": "대표전화"}, {"location": "1262,968,557,36", "text": "031-576-4545 출하실 031-576-3131"}, {"location": "403,1051,74,40", "text": "013"}, {"location": "1142,1052,528,71", "text": "타설완료• 0응 시 수"}, {"location": "289,1065,36,25", "text": "No"}, {"location": "287,1153,694,44", "text": "납 품 장 소 고덕 대림 아파트 현장"}, {"location": "663,1248,404,44", "text": "(175) 서울14다7478"}, {"location": "286,1252,194,37", "text": "운반차번호"}, {"location": "1439,1252,129,39", "text": "이은우"}, {"location": "858,1342,46,36", "text": "08"}, {"location": "1247,1342,45,36", "text": "03"}, {"location": "287,1392,194,36", "text": "납 품 시 각"}, {"location": "543,1439,133,36", "text": "도 착"}, {"location": "751,1522,91,37", "text": "6.00"}, {"location": "1547,1524,91,36", "text": "6.00"}, {"location": "287,1525,191,37", "text": "납 품 용 적"}, {"location": "1185,1529,31,35", "text": "계"}, {"location": "1762,1531,34,35", "text": "m3"}, {"location": "559,1600,423,33", "text": "콘크리트의 굵은골재의 최대"}, {"location": "1277,1603,171,31", "text": "슬럼프 또는"}, {"location": "1552,1604,199,32", "text": "시멘트 종류에"}, {"location": "1056,1623,129,32", "text": "호칭강도"}, {"location": "536,1644,449,33", "text": "종류에 따른 구분 치수에 따른 구분"}, {"location": "1276,1646,172,31", "text": "슬럼프 플로"}, {"location": "1585,1648,137,31", "text": "따른 구분"}, {"location": "284,1662,193,37", "text": "호 칭 방 법"}, {"location": "1496,1697,309,31", "text": "포를랜드시멘트 1종"}, {"location": "535,1711,211,31", "text": "보통콘크리트"}, {"location": "854,1714,374,37", "text": "25 mm 18 ,••曲"}, {"location": "1324,1714,146,37", "text": "150 mm"}, {"location": "877,1799,323,41", "text": "시방 배합표(kg/m3)"}, {"location": "258,1877,181,28", "text": "시멘트 시멘트"}, {"location": "551,1878,1267,36", "text": "회수수 잔골재 잔골재 산골재 굵은골재굵은골재 굵은골재 혼화재 혼화재 혼화재 혼화제 혼화제 혼화제"}, {"location": "480,1887,26,26", "text": "물"}, {"location": "683,1916,322,26", "text": "0 ㉣ ㉭ 0"}, {"location": "1278,1917,511,25", "text": "0 ㉣ ㉭ ㉭"}, {"location": "1178,1918,23,24", "text": "㉭"}, {"location": "691,2017,149,25", "text": "467 467"}, {"location": "992,2017,28,25", "text": "91"}, {"location": "1302,2017,31,25", "text": "43"}, {"location": "1569,2017,46,25", "text": "2.0"}, {"location": "269,2133,223,36", "text": "물결합재비"}, {"location": "742,2135,82,32", "text": "59.6"}, {"location": "1070,2135,197,36", "text": "산 골 재 율"}, {"location": "1482,2135,82,33", "text": "51.2"}, {"location": "1790,2140,26,27", "text": "0/0"}, {"location": "1340,2218,121,67", "text": "7%한"}, {"location": "269,2225,227,37", "text": "지 정 사 항"}, {"location": "1453,2311,260,40", "text": "4.5 士 15%"}, {"location": "544,2314,418,42", "text": "염화물량 : 0.3kg/m3 이하"}, {"location": "1233,2317,124,37", "text": "공기랑•"}, {"location": "460,2320,33,26", "text": "고"}, {"location": "1067,2386,198,36", "text": "출하계 확인"}, {"location": "1442,2406,120,36", "text": "인 성훈"}, {"location": "274,2407,213,36", "text": "인수자 확인"}, {"location": "1058,2430,216,36", "text": "표시사항확인"}, {"location": "540,2490,993,42", "text": "(성유보강제)상일동역사거리 직진 3-7게이트 329동"}, {"location": "462,2498,31,36", "text": "타"}, {"location": "967,2616,289,47", "text": "주식회사 산하"}, {"location": "249,2648,268,28", "text": "B5(182mm ※257mm)"}]')
    # 레이블 분리 모듈 - 임교진
    ocrData = splitLabel(ocrData)
    # ocrData = json.loads('[{"location": "1853,269,84,54", "text": "曇."}, {"location": "267,377,199,66", "text": "1-174"}, {"location": "643,453,591,66", "text": "레디 리스트 콘크리트"}, {"location": "1274,455,176,64", "text": "납품서"}, {"location": "1312,554,163,63", "text": "4體."}, {"location": "280,649,40,34", "text": "0"}, {"location": "320,649,120,34", "text": "표준명"}, {"location": "440,649,400,34", "text": ":레디믹스트콘크리트"}, {"location": "1342,652,379,32", "text": "132-81-13908"}, {"location": "1090,667,144,36", "text": "등록번호"}, {"location": "280,690,410,32", "text": "0표 준 번 호 . KS F 4009"}, {"location": "1027,720,207,48", "text": "亐 상 호"}, {"location": "1341,722,364,51", "text": "주식회사 é!0"}, {"location": "1683,729,179,75", "text": "•貳&譬4"}, {"location": "280,731,480,32", "text": "0인 증 번 호 . 제96-03-026호"}, {"location": "280,772,431,33", "text": "0인 증 기 관 : 한국표준협회"}, {"location": "1274,803,406,44", "text": "대표이사 전 찬 7"}, {"location": "1026,810,205,39", "text": "급 성 명"}, {"location": "280,813,643,35", "text": "0인| 증 종 류 : 보통,포장,고강도 콘크리트"}, {"location": "350,883,496,42", "text": "20 18년 11 월 08 일"}, {"location": "1273,885,475,39", "text": "경기도 남양주시 와부옵 수레로"}, {"location": "1199,890,34,30", "text": "소"}, {"location": "1026,916,35,36", "text": "자"}, {"location": "880,940,74,41", "text": "귀하"}, {"location": "288,941,287,48", "text": "대림산업(주)"}, {"location": "1090,965,146,35", "text": "대표전화"}, {"location": "1262,968,557,36", "text": "031-576-4545 출하실 031-576-3131"}, {"location": "403,1051,74,40", "text": "013"}, {"location": "1142,1052,528,71", "text": "타설완료• 0응 시 수"}, {"location": "289,1065,36,25", "text": "No"}, {"location": "287,1153,694,44", "text": "납 품 장 소 고덕 대림 아파트 현장"}, {"location": "663,1248,404,44", "text": "(175) 서울14다7478"}, {"location": "286,1252,194,37", "text": "운반차번호"}, {"location": "1439,1252,129,39", "text": "이은우"}, {"location": "858,1342,46,36", "text": "08"}, {"location": "1247,1342,45,36", "text": "03"}, {"location": "287,1392,194,36", "text": "납 품 시 각"}, {"location": "543,1439,133,36", "text": "도 착"}, {"location": "751,1522,91,37", "text": "6.00"}, {"location": "1547,1524,91,36", "text": "6.00"}, {"location": "287,1525,191,37", "text": "납 품 용 적"}, {"location": "1185,1529,31,35", "text": "계"}, {"location": "1762,1531,34,35", "text": "m3"}, {"location": "559,1600,180,33", "text": "콘크리트의"}, {"location": "739,1600,144,33", "text": "굵은골재"}, {"location": "883,1600,108,33", "text": "의최대"}, {"location": "1277,1603,171,31", "text": "슬럼프 또는"}, {"location": "1552,1604,102,32", "text": "시멘트"}, {"location": "1654,1604,102,32", "text": "종류에"}, {"location": "1056,1623,129,32", "text": "호칭강도"}, {"location": "536,1644,231,33", "text": "종류에따른구분"}, {"location": "767,1644,231,33", "text": "치수에따른구분"}, {"location": "1276,1646,172,31", "text": "슬럼프 플로"}, {"location": "1585,1648,137,31", "text": "따른 구분"}, {"location": "284,1662,193,37", "text": "호 칭 방 법"}, {"location": "1496,1697,140,31", "text": "포를랜드"}, {"location": "1636,1697,105,31", "text": "시멘트"}, {"location": "1741,1697,70,31", "text": "1종"}, {"location": "535,1711,211,31", "text": "보통콘크리트"}, {"location": "854,1714,374,37", "text": "25 mm 18 ,••曲"}, {"location": "1324,1714,146,37", "text": "150 mm"}, {"location": "877,1799,323,41", "text": "시방 배합표(kg/m3)"}, {"location": "258,1877,93,28", "text": "시멘트"}, {"location": "351,1877,93,28", "text": "시멘트"}, {"location": "551,1878,93,36", "text": "회수수"}, {"location": "644,1878,93,36", "text": "잔골재"}, {"location": "737,1878,93,36", "text": "잔골재"}, {"location": "830,1878,93,36", "text": "산골재"}, {"location": "923,1878,124,36", "text": "굵은골재"}, {"location": "1047,1878,124,36", "text": "굵은골재"}, {"location": "1171,1878,124,36", "text": "굵은골재"}, {"location": "1295,1878,93,36", "text": "혼화재"}, {"location": "1388,1878,93,36", "text": "혼화재"}, {"location": "1481,1878,93,36", "text": "혼화재"}, {"location": "1574,1878,93,36", "text": "혼화제"}, {"location": "1667,1878,93,36", "text": "혼화제"}, {"location": "1760,1878,93,36", "text": "혼화제"}, {"location": "480,1887,26,26", "text": "물"}, {"location": "683,1916,322,26", "text": "0 ㉣ ㉭ 0"}, {"location": "1278,1917,511,25", "text": "0 ㉣ ㉭ ㉭"}, {"location": "1178,1918,23,24", "text": "㉭"}, {"location": "691,2017,149,25", "text": "467 467"}, {"location": "992,2017,28,25", "text": "91"}, {"location": "1302,2017,31,25", "text": "43"}, {"location": "1569,2017,46,25", "text": "2.0"}, {"location": "269,2133,45,36", "text": "물"}, {"location": "314,2133,180,36", "text": "결합재비"}, {"location": "742,2135,82,32", "text": "59.6"}, {"location": "1070,2135,197,36", "text": "산 골 재 율"}, {"location": "1482,2135,82,33", "text": "51.2"}, {"location": "1790,2140,26,27", "text": "0/0"}, {"location": "1340,2218,121,67", "text": "7%한"}, {"location": "269,2225,227,37", "text": "지 정 사 항"}, {"location": "1453,2311,260,40", "text": "4.5 士 15%"}, {"location": "544,2314,56,42", "text": "염화"}, {"location": "600,2314,28,42", "text": "물"}, {"location": "628,2314,336,42", "text": "량:0.3kg/m3이하"}, {"location": "1233,2317,124,37", "text": "공기랑•"}, {"location": "460,2320,33,26", "text": "고"}, {"location": "1067,2386,198,36", "text": "출하계 확인"}, {"location": "1442,2406,120,36", "text": "인 성훈"}, {"location": "274,2407,213,36", "text": "인수자 확인"}, {"location": "1058,2430,216,36", "text": "표시사항확인"}, {"location": "540,2490,993,42", "text": "(성유보강제)상일동역사거리 직진 3-7게이트 329동"}, {"location": "462,2498,31,36", "text": "타"}, {"location": "967,2616,289,47", "text": "주식회사 산하"}, {"location": "249,2648,268,28", "text": "B5(182mm ※257mm)"}]')
    # doctype 추출 similarity - 임교진
    # docTopType, docType, maxNum = findDocType(ocrData)
    docTopType, docType, maxNum = 50, 70, 0.40236686390532544
    # Y축 데이터 X축 데이터 추출
    # ocrData = compareLabel(ocrData)
    # ocrData = extractCNNData(ocrData)
    # CNN Data
    # ocrData = json.loads('[{"location": "1853,269,84,54", "text": "曇.", "cnnData": "曇."}, {"location": "267,377,199,66", "text": "1-174", "cnnData": "1-174 시멘트 물 지정사항 인수자확인"}, {"location": "643,453,591,66", "text": "레디 리스트 콘크리트", "cnnData": "레디리스트콘크리트 납품서 잔골재"}, {"location": "1274,455,176,64", "text": "납품서", "cnnData": "납품서 대표이사전찬7 경기도남양주시와부옵수레로 슬럼프또는 슬럼프플로"}, {"location": "1312,554,163,63", "text": "4體.", "cnnData": "4體."}, {"location": "280,649,40,34", "text": "0", "cnnData": "0 표준명 :레디믹스트콘크리트 132-81-13908 0표준번호.KSF4009 0인증번호.제96-03-026호 0인증기관:한국표준협회 0인|증종류:보통포장고강도콘크리트"}, {"location": "320,649,120,34", "text": "표준명", "cnnData": "표준명 :레디믹스트콘크리트 132-81-13908 결합재비"}, {"location": "440,649,400,34", "text": ":레디믹스트콘크리트", "cnnData": ":레디믹스트콘크리트 132-81-13908"}, {"location": "1342,652,379,32", "text": "132-81-13908", "cnnData": "132-81-13908 주식회사é!0 7%한"}, {"location": "1090,667,144,36", "text": "등록번호", "cnnData": "등록번호 대표전화"}, {"location": "280,690,410,32", "text": "0표 준 번 호 . KS F 4009", "cnnData": "0표준번호.KSF4009 0인증번호.제96-03-026호 0인증기관:한국표준협회 0인|증종류:보통포장고강도콘크리트 대림산업(주)"}, {"location": "1027,720,207,48", "text": "亐 상 호", "cnnData": "亐상호 주식회사é!0 •貳&譬4 급성명 자"}, {"location": "1341,722,364,51", "text": "주식회사 é!0", "cnnData": "주식회사é!0 •貳&譬4 7%한"}, {"location": "1683,729,179,75", "text": "•貳&譬4", "cnnData": "•貳&譬4"}, {"location": "280,731,480,32", "text": "0인 증 번 호 . 제96-03-026호", "cnnData": "0인증번호.제96-03-026호 주식회사é!0 •貳&譬4 0인증기관:한국표준협회 0인|증종류:보통포장고강도콘크리트 대림산업(주) No"}, {"location": "280,772,431,33", "text": "0인 증 기 관 : 한국표준협회", "cnnData": "0인증기관:한국표준협회 0인|증종류:보통포장고강도콘크리트 대림산업(주) No 납품장소고덕대림아파트현장"}, {"location": "1274,803,406,44", "text": "대표이사 전 찬 7", "cnnData": "대표이사전찬7 경기도남양주시와부옵수레로 슬럼프또는 슬럼프플로 0㉣㉭㉭"}, {"location": "1026,810,205,39", "text": "급 성 명", "cnnData": "급성명 대표이사전찬7 자"}, {"location": "280,813,643,35", "text": "0인| 증 종 류 : 보통,포장,고강도 콘크리트", "cnnData": "0인|증종류:보통포장고강도콘크리트 급성명 대림산업(주) No 납품장소고덕대림아파트현장 운반차번호"}, {"location": "350,883,496,42", "text": "20 18년 11 월 08 일", "cnnData": "2018년11월08일 경기도남양주시와부옵수레로 소 시멘트"}, {"location": "1273,885,475,39", "text": "경기도 남양주시 와부옵 수레로", "cnnData": "경기도남양주시와부옵수레로 슬럼프또는 슬럼프플로 0㉣㉭㉭"}, {"location": "1199,890,34,30", "text": "소", "cnnData": "소 경기도남양주시와부옵수레로"}, {"location": "1026,916,35,36", "text": "자", "cnnData": "자"}, {"location": "880,940,74,41", "text": "귀하", "cnnData": "귀하 의최대 시방배합표(kg/m3)"}, {"location": "288,941,287,48", "text": "대림산업(주)", "cnnData": "대림산업(주) 귀하 No 납품장소고덕대림아파트현장 운반차번호 납품시각"}, {"location": "1090,965,146,35", "text": "대표전화", "cnnData": "대표전화 031-576-4545출하실031-576-3131"}, {"location": "1262,968,557,36", "text": "031-576-4545 출하실 031-576-3131", "cnnData": "031-576-4545출하실031-576-3131"}, {"location": "403,1051,74,40", "text": "013", "cnnData": "013 타설완료•0응시수"}, {"location": "1142,1052,528,71", "text": "타설완료• 0응 시 수", "cnnData": "타설완료•0응시수"}, {"location": "289,1065,36,25", "text": "No", "cnnData": "No 납품장소고덕대림아파트현장 운반차번호 납품시각 납품용적"}, {"location": "287,1153,694,44", "text": "납 품 장 소 고덕 대림 아파트 현장", "cnnData": "납품장소고덕대림아파트현장 운반차번호 납품시각 납품용적 호칭방법"}, {"location": "663,1248,404,44", "text": "(175) 서울14다7478", "cnnData": "(175)서울14다7478 이은우"}, {"location": "286,1252,194,37", "text": "운반차번호", "cnnData": "운반차번호 (175)서울14다7478 이은우 납품시각 납품용적 호칭방법"}, {"location": "1439,1252,129,39", "text": "이은우", "cnnData": "이은우 인성훈"}, {"location": "858,1342,46,36", "text": "08", "cnnData": "08 03 25mm18••曲"}, {"location": "1247,1342,45,36", "text": "03", "cnnData": "03"}, {"location": "287,1392,194,36", "text": "납 품 시 각", "cnnData": "납품시각 납품용적 호칭방법"}, {"location": "543,1439,133,36", "text": "도 착", "cnnData": "도착 종류에따른구분 보통콘크리트 회수수 염화"}, {"location": "751,1522,91,37", "text": "6.00", "cnnData": "6.00 6.00 계 m3 59.6"}, {"location": "1547,1524,91,36", "text": "6.00", "cnnData": "6.00 m3 시멘트"}, {"location": "287,1525,191,37", "text": "납 품 용 적", "cnnData": "납품용적 6.00 6.00 계 m3 호칭방법"}, {"location": "1185,1529,31,35", "text": "계", "cnnData": "계 6.00 m3 ㉭"}, {"location": "1762,1531,34,35", "text": "m3", "cnnData": "m3 혼화제"}, {"location": "559,1600,180,33", "text": "콘크리트의", "cnnData": "콘크리트의 굵은골재 의최대 슬럼프또는 시멘트 회수수"}, {"location": "739,1600,144,33", "text": "굵은골재", "cnnData": "굵은골재 의최대 슬럼프또는 시멘트 종류에 잔골재 59.6"}, {"location": "883,1600,108,33", "text": "의최대", "cnnData": "의최대 슬럼프또는 시멘트 종류에 시방배합표(kg/m3)"}, {"location": "1277,1603,171,31", "text": "슬럼프 또는", "cnnData": "슬럼프또는 시멘트 종류에 슬럼프플로 0㉣㉭㉭"}, {"location": "1552,1604,102,32", "text": "시멘트", "cnnData": "시멘트 종류에"}, {"location": "1654,1604,102,32", "text": "종류에", "cnnData": "종류에"}, {"location": "1056,1623,129,32", "text": "호칭강도", "cnnData": "호칭강도 굵은골재 표시사항확인"}, {"location": "536,1644,231,33", "text": "종류에따른구분", "cnnData": "종류에따른구분 치수에따른구분 슬럼프플로 따른구분 보통콘크리트 염화 (성유보강제)상일동역사거리직진3-7게이트329동"}, {"location": "767,1644,231,33", "text": "치수에따른구분", "cnnData": "치수에따른구분 슬럼프플로 따른구분"}, {"location": "1276,1646,172,31", "text": "슬럼프 플로", "cnnData": "슬럼프플로 따른구분 0㉣㉭㉭"}, {"location": "1585,1648,137,31", "text": "따른 구분", "cnnData": "따른구분"}, {"location": "284,1662,193,37", "text": "호 칭 방 법", "cnnData": "호칭방법"}, {"location": "1496,1697,140,31", "text": "포를랜드", "cnnData": "포를랜드 시멘트 1종"}, {"location": "1636,1697,105,31", "text": "시멘트", "cnnData": "시멘트 1종"}, {"location": "1741,1697,70,31", "text": "1종", "cnnData": "1종"}, {"location": "535,1711,211,31", "text": "보통콘크리트", "cnnData": "보통콘크리트 25mm18••曲 150mm 염화 (성유보강제)상일동역사거리직진3-7게이트329동"}, {"location": "854,1714,374,37", "text": "25 mm 18 ,••曲", "cnnData": "25mm18••曲 150mm"}, {"location": "1324,1714,146,37", "text": "150 mm", "cnnData": "150mm"}, {"location": "877,1799,323,41", "text": "시방 배합표(kg/m3)", "cnnData": "시방배합표(kg/m3)"}, {"location": "258,1877,93,28", "text": "시멘트", "cnnData": "시멘트 시멘트 회수수 잔골재 잔골재 B5(182mm※257mm)"}, {"location": "351,1877,93,28", "text": "시멘트", "cnnData": "시멘트 회수수 잔골재 잔골재 산골재"}, {"location": "551,1878,93,36", "text": "회수수", "cnnData": "회수수 잔골재 잔골재 산골재 굵은골재 염화"}, {"location": "644,1878,93,36", "text": "잔골재", "cnnData": "잔골재 잔골재 산골재 굵은골재 굵은골재"}, {"location": "737,1878,93,36", "text": "잔골재", "cnnData": "잔골재 산골재 굵은골재 굵은골재 굵은골재 59.6"}, {"location": "830,1878,93,36", "text": "산골재", "cnnData": "산골재 굵은골재 굵은골재 굵은골재 혼화재"}, {"location": "923,1878,124,36", "text": "굵은골재", "cnnData": "굵은골재 굵은골재 굵은골재 혼화재 혼화재"}, {"location": "1047,1878,124,36", "text": "굵은골재", "cnnData": "굵은골재 굵은골재 혼화재 혼화재 혼화재"}, {"location": "1171,1878,124,36", "text": "굵은골재", "cnnData": "굵은골재 혼화재 혼화재 혼화재 혼화제 ㉭"}, {"location": "1295,1878,93,36", "text": "혼화재", "cnnData": "혼화재 혼화재 혼화재 혼화제 혼화제 43"}, {"location": "1388,1878,93,36", "text": "혼화재", "cnnData": "혼화재 혼화재 혼화제 혼화제 혼화제"}, {"location": "1481,1878,93,36", "text": "혼화재", "cnnData": "혼화재 혼화제 혼화제 혼화제 51.2"}, {"location": "1574,1878,93,36", "text": "혼화제", "cnnData": "혼화제 혼화제 혼화제 2.0"}, {"location": "1667,1878,93,36", "text": "혼화제", "cnnData": "혼화제 혼화제"}, {"location": "1760,1878,93,36", "text": "혼화제", "cnnData": "혼화제"}, {"location": "480,1887,26,26", "text": "물", "cnnData": "물 회수수 잔골재 잔골재 산골재"}, {"location": "683,1916,322,26", "text": "0 ㉣ ㉭ 0", "cnnData": "0㉣㉭0 0㉣㉭㉭ ㉭ 467467"}, {"location": "1278,1917,511,25", "text": "0 ㉣ ㉭ ㉭", "cnnData": "0㉣㉭㉭"}, {"location": "1178,1918,23,24", "text": "㉭", "cnnData": "㉭ 0㉣㉭㉭"}, {"location": "691,2017,149,25", "text": "467 467", "cnnData": "467467 91 43 2.0"}, {"location": "992,2017,28,25", "text": "91", "cnnData": "91 43 2.0"}, {"location": "1302,2017,31,25", "text": "43", "cnnData": "43 2.0"}, {"location": "1569,2017,46,25", "text": "2.0", "cnnData": "2.0"}, {"location": "269,2133,45,36", "text": "물", "cnnData": "물 결합재비 59.6 산골재율 51.2 지정사항 인수자확인"}, {"location": "314,2133,180,36", "text": "결합재비", "cnnData": "결합재비 59.6 산골재율 51.2 0/0"}, {"location": "742,2135,82,32", "text": "59.6", "cnnData": "59.6 산골재율 51.2 0/0"}, {"location": "1070,2135,197,36", "text": "산 골 재 율", "cnnData": "산골재율 51.2 0/0 출하계확인"}, {"location": "1482,2135,82,33", "text": "51.2", "cnnData": "51.2 0/0"}, {"location": "1790,2140,26,27", "text": "0/0", "cnnData": "0/0"}, {"location": "1340,2218,121,67", "text": "7%한", "cnnData": "7%한"}, {"location": "269,2225,227,37", "text": "지 정 사 항", "cnnData": "지정사항 7%한 인수자확인"}, {"location": "1453,2311,260,40", "text": "4.5 士 15%", "cnnData": "4.5士15%"}, {"location": "544,2314,56,42", "text": "염화", "cnnData": "염화 4.5士15% 물 량:0.3kg/m3이하 공기랑• (성유보강제)상일동역사거리직진3-7게이트329동"}, {"location": "600,2314,28,42", "text": "물", "cnnData": "물 4.5士15% 량:0.3kg/m3이하 공기랑•"}, {"location": "628,2314,336,42", "text": "량:0.3kg/m3이하", "cnnData": "량:0.3kg/m3이하 4.5士15% 공기랑•"}, {"location": "1233,2317,124,37", "text": "공기랑•", "cnnData": "공기랑• 4.5士15%"}, {"location": "460,2320,33,26", "text": "고", "cnnData": "고 4.5士15% 염화 물 량:0.3kg/m3이하 타"}, {"location": "1067,2386,198,36", "text": "출하계 확인", "cnnData": "출하계확인 표시사항확인"}, {"location": "1442,2406,120,36", "text": "인 성훈", "cnnData": "인성훈"}, {"location": "274,2407,213,36", "text": "인수자 확인", "cnnData": "인수자확인 인성훈"}, {"location": "1058,2430,216,36", "text": "표시사항확인", "cnnData": "표시사항확인"}, {"location": "540,2490,993,42", "text": "(성유보강제)상일동역사거리 직진 3-7게이트 329동", "cnnData": "(성유보강제)상일동역사거리직진3-7게이트329동"}, {"location": "462,2498,31,36", "text": "타", "cnnData": "타 (성유보강제)상일동역사거리직진3-7게이트329동"}, {"location": "967,2616,289,47", "text": "주식회사 산하", "cnnData": "주식회사산하"}, {"location": "249,2648,268,28", "text": "B5(182mm ※257mm)", "cnnData": "B5(182mm※257mm)"}]')
    # x축, y축 Data
    # ocrData = json.loads('[{"location": "1853,269,84,54", "text": "曇.", "xData": "曇.", "yData": "曇."}, {"location": "267,377,199,66", "text": "1-174", "xData": "1-174", "yData": "1-174 시멘트 물 지정사항 인수자확인"}, {"location": "643,453,591,66", "text": "레디 리스트 콘크리트", "xData": "레디리스트콘크리트 납품서", "yData": "레디리스트콘크리트 잔골재"}, {"location": "1274,455,176,64", "text": "납품서", "xData": "납품서", "yData": "납품서 대표이사전찬7 경기도남양주시와부옵수레로 슬럼프또는 슬럼프플로"}, {"location": "1312,554,163,63", "text": "4體.", "xData": "4體.", "yData": "4體."}, {"location": "280,649,40,34", "text": "0", "xData": "0 표준명 :레디믹스트콘크리트 132-81-13908", "yData": "0 0표준번호.KSF4009 0인증번호.제96-03-026호 0인증기관:한국표준협회 0인|증종류:보통,포장,고강도콘크리트"}, {"location": "320,649,120,34", "text": "표준명", "xData": "표준명 :레디믹스트콘크리트 132-81-13908", "yData": "표준명 결합재비"}, {"location": "440,649,400,34", "text": ":레디믹스트콘크리트", "xData": ":레디믹스트콘크리트 132-81-13908", "yData": ":레디믹스트콘크리트"}, {"location": "1342,652,379,32", "text": "132-81-13908", "xData": "132-81-13908", "yData": "132-81-13908 주식회사é!0 7%한"}, {"location": "1090,667,144,36", "text": "등록번호", "xData": "등록번호", "yData": "등록번호 대표전화"}, {"location": "280,690,410,32", "text": "0표 준 번 호 . KS F 4009", "xData": "0표준번호.KSF4009", "yData": "0표준번호.KSF4009 0인증번호.제96-03-026호 0인증기관:한국표준협회 0인|증종류:보통,포장,고강도콘크리트 대림산업(주)"}, {"location": "1027,720,207,48", "text": "亐 상 호", "xData": "亐상호 주식회사é!0 •貳&譬4", "yData": "亐상호 급성명 자"}, {"location": "1341,722,364,51", "text": "주식회사 é!0", "xData": "주식회사é!0 •貳&譬4", "yData": "주식회사é!0 7%한"}, {"location": "1683,729,179,75", "text": "•貳&譬4", "xData": "•貳&譬4", "yData": "•貳&譬4"}, {"location": "280,731,480,32", "text": "0인 증 번 호 . 제96-03-026호", "xData": "0인증번호.제96-03-026호 주식회사é!0 •貳&譬4", "yData": "0인증번호.제96-03-026호 0인증기관:한국표준협회 0인|증종류:보통,포장,고강도콘크리트 대림산업(주) No"}, {"location": "280,772,431,33", "text": "0인 증 기 관 : 한국표준협회", "xData": "0인증기관:한국표준협회", "yData": "0인증기관:한국표준협회 0인|증종류:보통,포장,고강도콘크리트 대림산업(주) No 납품장소고덕대림아파트현장"}, {"location": "1274,803,406,44", "text": "대표이사 전 찬 7", "xData": "대표이사전찬7", "yData": "대표이사전찬7 경기도남양주시와부옵수레로 슬럼프또는 슬럼프플로 0㉣㉭㉭"}, {"location": "1026,810,205,39", "text": "급 성 명", "xData": "급성명 대표이사전찬7", "yData": "급성명 자"}, {"location": "280,813,643,35", "text": "0인| 증 종 류 : 보통,포장,고강도 콘크리트", "xData": "0인|증종류:보통,포장,고강도콘크리트 급성명", "yData": "0인|증종류:보통,포장,고강도콘크리트 대림산업(주) No 납품장소고덕대림아파트현장 운반차번호"}, {"location": "350,883,496,42", "text": "20 18년 11 월 08 일", "xData": "2018년11월08일 경기도남양주시와부옵수레로 소", "yData": "2018년11월08일 시멘트"}, {"location": "1273,885,475,39", "text": "경기도 남양주시 와부옵 수레로", "xData": "경기도남양주시와부옵수레로", "yData": "경기도남양주시와부옵수레로 슬럼프또는 슬럼프플로 0㉣㉭㉭"}, {"location": "1199,890,34,30", "text": "소", "xData": "소 경기도남양주시와부옵수레로", "yData": "소"}, {"location": "1026,916,35,36", "text": "자", "xData": "자", "yData": "자"}, {"location": "880,940,74,41", "text": "귀하", "xData": "귀하", "yData": "귀하 의최대 시방배합표(kg/m3)"}, {"location": "288,941,287,48", "text": "대림산업(주)", "xData": "대림산업(주) 귀하", "yData": "대림산업(주) No 납품장소고덕대림아파트현장 운반차번호 납품시각"}, {"location": "1090,965,146,35", "text": "대표전화", "xData": "대표전화 031-576-4545출하실031-576-3131", "yData": "대표전화"}, {"location": "1262,968,557,36", "text": "031-576-4545 출하실 031-576-3131", "xData": "031-576-4545출하실031-576-3131", "yData": "031-576-4545출하실031-576-3131"}, {"location": "403,1051,74,40", "text": "013", "xData": "013 타설완료•0응시수", "yData": "013"}, {"location": "1142,1052,528,71", "text": "타설완료• 0응 시 수", "xData": "타설완료•0응시수", "yData": "타설완료•0응시수"}, {"location": "289,1065,36,25", "text": "No", "xData": "No", "yData": "No 납품장소고덕대림아파트현장 운반차번호 납품시각 납품용적"}, {"location": "287,1153,694,44", "text": "납 품 장 소 고덕 대림 아파트 현장", "xData": "납품장소고덕대림아파트현장", "yData": "납품장소고덕대림아파트현장 운반차번호 납품시각 납품용적 호칭방법"}, {"location": "663,1248,404,44", "text": "(175) 서울14다7478", "xData": "(175)서울14다7478 이은우", "yData": "(175)서울14다7478"}, {"location": "286,1252,194,37", "text": "운반차번호", "xData": "운반차번호 (175)서울14다7478 이은우", "yData": "운반차번호 납품시각 납품용적 호칭방법"}, {"location": "1439,1252,129,39", "text": "이은우", "xData": "이은우", "yData": "이은우 인성훈"}, {"location": "858,1342,46,36", "text": "08", "xData": "08 03", "yData": "08 25mm18,••曲"}, {"location": "1247,1342,45,36", "text": "03", "xData": "03", "yData": "03"}, {"location": "287,1392,194,36", "text": "납 품 시 각", "xData": "납품시각", "yData": "납품시각 납품용적 호칭방법"}, {"location": "543,1439,133,36", "text": "도 착", "xData": "도착", "yData": "도착 종류에따른구분 보통콘크리트 회수수 염화"}, {"location": "751,1522,91,37", "text": "6.00", "xData": "6.00 6.00 계 m3", "yData": "6.00 59.6"}, {"location": "1547,1524,91,36", "text": "6.00", "xData": "6.00 m3", "yData": "6.00 시멘트"}, {"location": "287,1525,191,37", "text": "납 품 용 적", "xData": "납품용적 6.00 6.00 계 m3", "yData": "납품용적 호칭방법"}, {"location": "1185,1529,31,35", "text": "계", "xData": "계 6.00 m3", "yData": "계 ㉭"}, {"location": "1762,1531,34,35", "text": "m3", "xData": "m3", "yData": "m3 혼화제"}, {"location": "559,1600,180,33", "text": "콘크리트의", "xData": "콘크리트의 굵은골재 의최대 슬럼프또는 시멘트", "yData": "콘크리트의 회수수"}, {"location": "739,1600,144,33", "text": "굵은골재", "xData": "굵은골재 의최대 슬럼프또는 시멘트 종류에", "yData": "굵은골재 잔골재 59.6"}, {"location": "883,1600,108,33", "text": "의최대", "xData": "의최대 슬럼프또는 시멘트 종류에", "yData": "의최대 시방배합표(kg/m3)"}, {"location": "1277,1603,171,31", "text": "슬럼프 또는", "xData": "슬럼프또는 시멘트 종류에", "yData": "슬럼프또는 슬럼프플로 0㉣㉭㉭"}, {"location": "1552,1604,102,32", "text": "시멘트", "xData": "시멘트 종류에", "yData": "시멘트"}, {"location": "1654,1604,102,32", "text": "종류에", "xData": "종류에", "yData": "종류에"}, {"location": "1056,1623,129,32", "text": "호칭강도", "xData": "호칭강도", "yData": "호칭강도 굵은골재 표시사항확인"}, {"location": "536,1644,231,33", "text": "종류에따른구분", "xData": "종류에따른구분 치수에따른구분 슬럼프플로 따른구분", "yData": "종류에따른구분 보통콘크리트 염화 (성유보강제)상일동역사거리직진3-7게이트329동"}, {"location": "767,1644,231,33", "text": "치수에따른구분", "xData": "치수에따른구분 슬럼프플로 따른구분", "yData": "치수에따른구분"}, {"location": "1276,1646,172,31", "text": "슬럼프 플로", "xData": "슬럼프플로 따른구분", "yData": "슬럼프플로 0㉣㉭㉭"}, {"location": "1585,1648,137,31", "text": "따른 구분", "xData": "따른구분", "yData": "따른구분"}, {"location": "284,1662,193,37", "text": "호 칭 방 법", "xData": "호칭방법", "yData": "호칭방법"}, {"location": "1496,1697,140,31", "text": "포를랜드", "xData": "포를랜드 시멘트 1종", "yData": "포를랜드"}, {"location": "1636,1697,105,31", "text": "시멘트", "xData": "시멘트 1종", "yData": "시멘트"}, {"location": "1741,1697,70,31", "text": "1종", "xData": "1종", "yData": "1종"}, {"location": "535,1711,211,31", "text": "보통콘크리트", "xData": "보통콘크리트 25mm18,••曲 150mm", "yData": "보통콘크리트 염화 (성유보강제)상일동역사거리직진3-7게이트329동"}, {"location": "854,1714,374,37", "text": "25 mm 18 ,••曲", "xData": "25mm18,••曲 150mm", "yData": "25mm18,••曲"}, {"location": "1324,1714,146,37", "text": "150 mm", "xData": "150mm", "yData": "150mm"}, {"location": "877,1799,323,41", "text": "시방 배합표(kg/m3)", "xData": "시방배합표(kg/m3)", "yData": "시방배합표(kg/m3)"}, {"location": "258,1877,93,28", "text": "시멘트", "xData": "시멘트 시멘트 회수수 잔골재 잔골재", "yData": "시멘트 B5(182mm※257mm)"}, {"location": "351,1877,93,28", "text": "시멘트", "xData": "시멘트 회수수 잔골재 잔골재 산골재", "yData": "시멘트"}, {"location": "551,1878,93,36", "text": "회수수", "xData": "회수수 잔골재 잔골재 산골재 굵은골재", "yData": "회수수 염화"}, {"location": "644,1878,93,36", "text": "잔골재", "xData": "잔골재 잔골재 산골재 굵은골재 굵은골재", "yData": "잔골재"}, {"location": "737,1878,93,36", "text": "잔골재", "xData": "잔골재 산골재 굵은골재 굵은골재 굵은골재", "yData": "잔골재 59.6"}, {"location": "830,1878,93,36", "text": "산골재", "xData": "산골재 굵은골재 굵은골재 굵은골재 혼화재", "yData": "산골재"}, {"location": "923,1878,124,36", "text": "굵은골재", "xData": "굵은골재 굵은골재 굵은골재 혼화재 혼화재", "yData": "굵은골재"}, {"location": "1047,1878,124,36", "text": "굵은골재", "xData": "굵은골재 굵은골재 혼화재 혼화재 혼화재", "yData": "굵은골재"}, {"location": "1171,1878,124,36", "text": "굵은골재", "xData": "굵은골재 혼화재 혼화재 혼화재 혼화제", "yData": "굵은골재 ㉭"}, {"location": "1295,1878,93,36", "text": "혼화재", "xData": "혼화재 혼화재 혼화재 혼화제 혼화제", "yData": "혼화재 43"}, {"location": "1388,1878,93,36", "text": "혼화재", "xData": "혼화재 혼화재 혼화제 혼화제 혼화제", "yData": "혼화재"}, {"location": "1481,1878,93,36", "text": "혼화재", "xData": "혼화재 혼화제 혼화제 혼화제", "yData": "혼화재 51.2"}, {"location": "1574,1878,93,36", "text": "혼화제", "xData": "혼화제 혼화제 혼화제", "yData": "혼화제 2.0"}, {"location": "1667,1878,93,36", "text": "혼화제", "xData": "혼화제 혼화제", "yData": "혼화제"}, {"location": "1760,1878,93,36", "text": "혼화제", "xData": "혼화제", "yData": "혼화제"}, {"location": "480,1887,26,26", "text": "물", "xData": "물 회수수 잔골재 잔골재 산골재", "yData": "물"}, {"location": "683,1916,322,26", "text": "0 ㉣ ㉭ 0", "xData": "0㉣㉭0 0㉣㉭㉭ ㉭", "yData": "0㉣㉭0 467467"}, {"location": "1278,1917,511,25", "text": "0 ㉣ ㉭ ㉭", "xData": "0㉣㉭㉭", "yData": "0㉣㉭㉭"}, {"location": "1178,1918,23,24", "text": "㉭", "xData": "㉭ 0㉣㉭㉭", "yData": "㉭"}, {"location": "691,2017,149,25", "text": "467 467", "xData": "467467 91 43 2.0", "yData": "467467"}, {"location": "992,2017,28,25", "text": "91", "xData": "91 43 2.0", "yData": "91"}, {"location": "1302,2017,31,25", "text": "43", "xData": "43 2.0", "yData": "43"}, {"location": "1569,2017,46,25", "text": "2.0", "xData": "2.0", "yData": "2.0"}, {"location": "269,2133,45,36", "text": "물", "xData": "물 결합재비 59.6 산골재율 51.2", "yData": "물 지정사항 인수자확인"}, {"location": "314,2133,180,36", "text": "결합재비", "xData": "결합재비 59.6 산골재율 51.2 0/0", "yData": "결합재비"}, {"location": "742,2135,82,32", "text": "59.6", "xData": "59.6 산골재율 51.2 0/0", "yData": "59.6"}, {"location": "1070,2135,197,36", "text": "산 골 재 율", "xData": "산골재율 51.2 0/0", "yData": "산골재율 출하계확인"}, {"location": "1482,2135,82,33", "text": "51.2", "xData": "51.2 0/0", "yData": "51.2"}, {"location": "1790,2140,26,27", "text": "0/0", "xData": "0/0", "yData": "0/0"}, {"location": "1340,2218,121,67", "text": "7%한", "xData": "7%한", "yData": "7%한"}, {"location": "269,2225,227,37", "text": "지 정 사 항", "xData": "지정사항 7%한", "yData": "지정사항 인수자확인"}, {"location": "1453,2311,260,40", "text": "4.5 士 15%", "xData": "4.5士15%", "yData": "4.5士15%"}, {"location": "544,2314,56,42", "text": "염화", "xData": "염화 4.5士15% 물 량:0.3kg/m3이하 공기랑•", "yData": "염화 (성유보강제)상일동역사거리직진3-7게이트329동"}, {"location": "600,2314,28,42", "text": "물", "xData": "물 4.5士15% 량:0.3kg/m3이하 공기랑•", "yData": "물"}, {"location": "628,2314,336,42", "text": "량:0.3kg/m3이하", "xData": "량:0.3kg/m3이하 4.5士15% 공기랑•", "yData": "량:0.3kg/m3이하"}, {"location": "1233,2317,124,37", "text": "공기랑•", "xData": "공기랑• 4.5士15%", "yData": "공기랑•"}, {"location": "460,2320,33,26", "text": "고", "xData": "고 4.5士15% 염화 물 량:0.3kg/m3이하", "yData": "고 타"}, {"location": "1067,2386,198,36", "text": "출하계 확인", "xData": "출하계확인", "yData": "출하계확인 표시사항확인"}, {"location": "1442,2406,120,36", "text": "인 성훈", "xData": "인성훈", "yData": "인성훈"}, {"location": "274,2407,213,36", "text": "인수자 확인", "xData": "인수자확인 인성훈", "yData": "인수자확인"}, {"location": "1058,2430,216,36", "text": "표시사항확인", "xData": "표시사항확인", "yData": "표시사항확인"}, {"location": "540,2490,993,42", "text": "(성유보강제)상일동역사거리 직진 3-7게이트 329동", "xData": "(성유보강제)상일동역사거리직진3-7게이트329동", "yData": "(성유보강제)상일동역사거리직진3-7게이트329동"}, {"location": "462,2498,31,36", "text": "타", "xData": "타 (성유보강제)상일동역사거리직진3-7게이트329동", "yData": "타"}, {"location": "967,2616,289,47", "text": "주식회사 산하", "xData": "주식회사산하", "yData": "주식회사산하"}, {"location": "249,2648,268,28", "text": "B5(182mm ※257mm)", "xData": "B5(182mm※257mm)", "yData": "B5(182mm※257mm)"}]')
    # label 추출 MS ML 호출
    # labelData = findColByML(ocrData)
    # entry 추출
    # entryData = findColByML(ocrData)

    '''
    {'location': '1853,269,84,54', 'text': '曇.', 'cnnData': '曇. ', 'colLbl': '768'}
    {'location': '267,377,199,66', 'text': '1-174', 'cnnData': '1-174 시멘트 물 지정사항 인수자확인', 'colLbl': '768'}
    {'location': '643,453,591,66', 'text': '레디 리스트 콘크리트', 'cnnData': '레디리스트콘크리트 납품서 잔골재', 'colLbl': '768'}
    {'location': '1274,455,176,64', 'text': '납품서', 'cnnData': '납품서 대표이사전찬7 경기도남양주시와부옵수레로 슬럼프또는 슬럼프플로', 'colLbl': '768'}
    {'location': '1312,554,163,63', 'text': '4體.', 'cnnData': '4體. ', 'colLbl': '768'}
    {'location': '280,649,40,34', 'text': '0', 'cnnData': '0 표준명 :레디믹스트콘크리트 132-81-13908 0표준번호.KSF4009 0인증번호.제96-03-026호 0인증기관:한국표준협회 0인|증종류:보통포장고강도콘크리트', 'colLbl': '768'}
    {'location': '320,649,120,34', 'text': '표준명', 'cnnData': '표준명 :레디믹스트콘크리트 132-81-13908 결합재비', 'colLbl': '768'}
    {'location': '440,649,400,34', 'text': ':레디믹스트콘크리트', 'cnnData': ':레디믹스트콘크리트 132-81-13908 ', 'colLbl': '768'}
    {'location': '1342,652,379,32', 'text': '132-81-13908', 'cnnData': '132-81-13908 주식회사é!0 7%한', 'colLbl': '768'}
    {'location': '1090,667,144,36', 'text': '등록번호', 'cnnData': '등록번호 대표전화', 'colLbl': '768'}
    {'location': '280,690,410,32', 'text': '0표 준 번 호 . KS F 4009', 'cnnData': '0표준번호.KSF4009 0인증번호.제96-03-026호 0인증기관:한국표준협회 0인|증종류:보통포장고강도콘크리트 대림산업(주)', 'colLbl': '768'}
    {'location': '1027,720,207,48', 'text': '亐 상 호', 'cnnData': '亐상호 주식회사é!0 •貳&譬4 급성명 자', 'colLbl': '768'}
    {'location': '1341,722,364,51', 'text': '주식회사 é!0', 'cnnData': '주식회사é!0 •貳&譬4 7%한', 'colLbl': '768'}
    {'location': '1683,729,179,75', 'text': '•貳&譬4', 'cnnData': '•貳&譬4 ', 'colLbl': '768'}
    {'location': '280,731,480,32', 'text': '0인 증 번 호 . 제96-03-026호', 'cnnData': '0인증번호.제96-03-026호 주식회사é!0 •貳&譬4 0인증기관:한국표준협회 0인|증종류:보통포장고강도콘크리트 대림산업(주) No', 'colLbl': '768'}
    {'location': '280,772,431,33', 'text': '0인 증 기 관 : 한국표준협회', 'cnnData': '0인증기관:한국표준협회 0인|증종류:보통포장고강도콘크리트 대림산업(주) No 납품장소고덕대림아파트현장', 'colLbl': '768'}
    {'location': '1274,803,406,44', 'text': '대표이사 전 찬 7', 'cnnData': '대표이사전찬7 경기도남양주시와부옵수레로 슬럼프또는 슬럼프플로 0㉣㉭㉭', 'colLbl': '768'}
    {'location': '1026,810,205,39', 'text': '급 성 명', 'cnnData': '급성명 대표이사전찬7 자', 'colLbl': '768'}
    {'location': '280,813,643,35', 'text': '0인| 증 종 류 : 보통,포장,고강도 콘크리트', 'cnnData': '0인|증종류:보통포장고강도콘크리트 급성명 대림산업(주) No 납품장소고덕대림아파트현장 운반차번호', 'colLbl': '768'}
    {'location': '350,883,496,42', 'text': '20 18년 11 월 08 일', 'cnnData': '2018년11월08일 경기도남양주시와부옵수레로 소 시멘트', 'colLbl': '768'}
    {'location': '1273,885,475,39', 'text': '경기도 남양주시 와부옵 수레로', 'cnnData': '경기도남양주시와부옵수레로 슬럼프또는 슬럼프플로 0㉣㉭㉭', 'colLbl': '768'}
    {'location': '1199,890,34,30', 'text': '소', 'cnnData': '소 경기도남양주시와부옵수레로 ', 'colLbl': '768'}
    {'location': '1026,916,35,36', 'text': '자', 'cnnData': '자 ', 'colLbl': '768'}
    {'location': '880,940,74,41', 'text': '귀하', 'cnnData': '귀하 의최대 시방배합표(kg/m3)', 'colLbl': '768'}
    {'location': '288,941,287,48', 'text': '대림산업(주)', 'cnnData': '대림산업(주) 귀하 No 납품장소고덕대림아파트현장 운반차번호 납품시각', 'colLbl': '768'}
    {'location': '1090,965,146,35', 'text': '대표전화', 'cnnData': '대표전화 031-576-4545출하실031-576-3131 ', 'colLbl': '768'}
    {'location': '1262,968,557,36', 'text': '031-576-4545 출하실 031-576-3131', 'cnnData': '031-576-4545출하실031-576-3131 ', 'colLbl': '768'}
    {'location': '403,1051,74,40', 'text': '013', 'cnnData': '013 타설완료•0응시수 ', 'colLbl': '768'}
    {'location': '1142,1052,528,71', 'text': '타설완료• 0응 시 수', 'cnnData': '타설완료•0응시수 ', 'colLbl': '768'}
    {'location': '289,1065,36,25', 'text': 'No', 'cnnData': 'No 납품장소고덕대림아파트현장 운반차번호 납품시각 납품용적', 'colLbl': '792'}
    {'location': '287,1153,694,44', 'text': '납 품 장 소 고덕 대림 아파트 현장', 'cnnData': '납품장소고덕대림아파트현장 운반차번호 납품시각 납품용적 호칭방법', 'colLbl': '768'}
    {'location': '663,1248,404,44', 'text': '(175) 서울14다7478', 'cnnData': '(175)서울14다7478 이은우 ', 'colLbl': '768'}
    {'location': '286,1252,194,37', 'text': '운반차번호', 'cnnData': '운반차번호 (175)서울14다7478 이은우 납품시각 납품용적 호칭방법', 'colLbl': '765'}
    {'location': '1439,1252,129,39', 'text': '이은우', 'cnnData': '이은우 인성훈', 'colLbl': '768'}
    {'location': '858,1342,46,36', 'text': '08', 'cnnData': '08 03 25mm18••曲', 'colLbl': '768'}
    {'location': '1247,1342,45,36', 'text': '03', 'cnnData': '03 ', 'colLbl': '768'}
    {'location': '287,1392,194,36', 'text': '납 품 시 각', 'cnnData': '납품시각 납품용적 호칭방법', 'colLbl': '768'}
    {'location': '543,1439,133,36', 'text': '도 착', 'cnnData': '도착 종류에따른구분 보통콘크리트 회수수 염화', 'colLbl': '768'}
    {'location': '751,1522,91,37', 'text': '6.00', 'cnnData': '6.00 6.00 계 m3 59.6', 'colLbl': '768'}
    {'location': '1547,1524,91,36', 'text': '6.00', 'cnnData': '6.00 m3 시멘트', 'colLbl': '768'}
    {'location': '287,1525,191,37', 'text': '납 품 용 적', 'cnnData': '납품용적 6.00 6.00 계 m3 호칭방법', 'colLbl': '768'}
    {'location': '1185,1529,31,35', 'text': '계', 'cnnData': '계 6.00 m3 ㉭', 'colLbl': '768'}
    {'location': '1762,1531,34,35', 'text': 'm3', 'cnnData': 'm3 혼화제', 'colLbl': '768'}
    {'location': '559,1600,180,33', 'text': '콘크리트의', 'cnnData': '콘크리트의 굵은골재 의최대 슬럼프또는 시멘트 회수수', 'colLbl': '768'}
    {'location': '739,1600,144,33', 'text': '굵은골재', 'cnnData': '굵은골재 의최대 슬럼프또는 시멘트 종류에 잔골재 59.6', 'colLbl': '783'}
    {'location': '883,1600,108,33', 'text': '의최대', 'cnnData': '의최대 슬럼프또는 시멘트 종류에 시방배합표(kg/m3)', 'colLbl': '768'}
    {'location': '1277,1603,171,31', 'text': '슬럼프 또는', 'cnnData': '슬럼프또는 시멘트 종류에 슬럼프플로 0㉣㉭㉭', 'colLbl': '768'}
    {'location': '1552,1604,102,32', 'text': '시멘트', 'cnnData': '시멘트 종류에 ', 'colLbl': '775'}
    {'location': '1654,1604,102,32', 'text': '종류에', 'cnnData': '종류에 ', 'colLbl': '768'}
    {'location': '1056,1623,129,32', 'text': '호칭강도', 'cnnData': '호칭강도 굵은골재 표시사항확인', 'colLbl': '771'}
    {'location': '536,1644,231,33', 'text': '종류에따른구분', 'cnnData': '종류에따른구분 치수에따른구분 슬럼프플로 따른구분 보통콘크리트 염화 (성유보강제)상일동역사거리직진3-7게이트329동', 'colLbl': '768'}
    {'location': '767,1644,231,33', 'text': '치수에따른구분', 'cnnData': '치수에따른구분 슬럼프플로 따른구분 ', 'colLbl': '768'}
    {'location': '1276,1646,172,31', 'text': '슬럼프 플로', 'cnnData': '슬럼프플로 따른구분 0㉣㉭㉭', 'colLbl': '768'}
    {'location': '1585,1648,137,31', 'text': '따른 구분', 'cnnData': '따른구분 ', 'colLbl': '768'}
    {'location': '284,1662,193,37', 'text': '호 칭 방 법', 'cnnData': '호칭방법 ', 'colLbl': '768'}
    {'location': '1496,1697,140,31', 'text': '포를랜드', 'cnnData': '포를랜드 시멘트 1종 ', 'colLbl': '768'}
    {'location': '1636,1697,105,31', 'text': '시멘트', 'cnnData': '시멘트 1종 ', 'colLbl': '775'}
    {'location': '1741,1697,70,31', 'text': '1종', 'cnnData': '1종 ', 'colLbl': '768'}
    {'location': '535,1711,211,31', 'text': '보통콘크리트', 'cnnData': '보통콘크리트 25mm18••曲 150mm 염화 (성유보강제)상일동역사거리직진3-7게이트329동', 'colLbl': '768'}
    {'location': '854,1714,374,37', 'text': '25 mm 18 ,••曲', 'cnnData': '25mm18••曲 150mm ', 'colLbl': '768'}
    {'location': '1324,1714,146,37', 'text': '150 mm', 'cnnData': '150mm ', 'colLbl': '768'}
    {'location': '877,1799,323,41', 'text': '시방 배합표(kg/m3)', 'cnnData': '시방배합표(kg/m3) ', 'colLbl': '768'}
    {'location': '258,1877,93,28', 'text': '시멘트', 'cnnData': '시멘트 시멘트 회수수 잔골재 잔골재 B5(182mm※257mm)', 'colLbl': '775'}
    {'location': '351,1877,93,28', 'text': '시멘트', 'cnnData': '시멘트 회수수 잔골재 잔골재 산골재 ', 'colLbl': '775'}
    {'location': '551,1878,93,36', 'text': '회수수', 'cnnData': '회수수 잔골재 잔골재 산골재 굵은골재 염화', 'colLbl': '777'}
    {'location': '644,1878,93,36', 'text': '잔골재', 'cnnData': '잔골재 잔골재 산골재 굵은골재 굵은골재 ', 'colLbl': '780'}
    {'location': '737,1878,93,36', 'text': '잔골재', 'cnnData': '잔골재 산골재 굵은골재 굵은골재 굵은골재 59.6', 'colLbl': '780'}
    {'location': '830,1878,93,36', 'text': '산골재', 'cnnData': '산골재 굵은골재 굵은골재 굵은골재 혼화재 ', 'colLbl': '768'}
    {'location': '923,1878,124,36', 'text': '굵은골재', 'cnnData': '굵은골재 굵은골재 굵은골재 혼화재 혼화재 ', 'colLbl': '783'}
    {'location': '1047,1878,124,36', 'text': '굵은골재', 'cnnData': '굵은골재 굵은골재 혼화재 혼화재 혼화재 ', 'colLbl': '783'}
    {'location': '1171,1878,124,36', 'text': '굵은골재', 'cnnData': '굵은골재 혼화재 혼화재 혼화재 혼화제 ㉭', 'colLbl': '783'}
    {'location': '1295,1878,93,36', 'text': '혼화재', 'cnnData': '혼화재 혼화재 혼화재 혼화제 혼화제 43', 'colLbl': '786'}
    {'location': '1388,1878,93,36', 'text': '혼화재', 'cnnData': '혼화재 혼화재 혼화제 혼화제 혼화제 ', 'colLbl': '786'}
    {'location': '1481,1878,93,36', 'text': '혼화재', 'cnnData': '혼화재 혼화제 혼화제 혼화제 51.2', 'colLbl': '786'}
    {'location': '1574,1878,93,36', 'text': '혼화제', 'cnnData': '혼화제 혼화제 혼화제 2.0', 'colLbl': '789'}
    {'location': '1667,1878,93,36', 'text': '혼화제', 'cnnData': '혼화제 혼화제 ', 'colLbl': '789'}
    {'location': '1760,1878,93,36', 'text': '혼화제', 'cnnData': '혼화제 ', 'colLbl': '789'}
    {'location': '480,1887,26,26', 'text': '물', 'cnnData': '물 회수수 잔골재 잔골재 산골재 ', 'colLbl': '776'}
    {'location': '683,1916,322,26', 'text': '0 ㉣ ㉭ 0', 'cnnData': '0㉣㉭0 0㉣㉭㉭ ㉭ 467467', 'colLbl': '768'}
    {'location': '1278,1917,511,25', 'text': '0 ㉣ ㉭ ㉭', 'cnnData': '0㉣㉭㉭ ', 'colLbl': '768'}
    {'location': '1178,1918,23,24', 'text': '㉭', 'cnnData': '㉭ 0㉣㉭㉭ ', 'colLbl': '768'}
    {'location': '691,2017,149,25', 'text': '467 467', 'cnnData': '467467 91 43 2.0 ', 'colLbl': '768'}
    {'location': '992,2017,28,25', 'text': '91', 'cnnData': '91 43 2.0 ', 'colLbl': '768'}
    {'location': '1302,2017,31,25', 'text': '43', 'cnnData': '43 2.0 ', 'colLbl': '768'}
    {'location': '1569,2017,46,25', 'text': '2.0', 'cnnData': '2.0 ', 'colLbl': '768'}
    {'location': '269,2133,45,36', 'text': '물', 'cnnData': '물 결합재비 59.6 산골재율 51.2 지정사항 인수자확인', 'colLbl': '776'}
    {'location': '314,2133,180,36', 'text': '결합재비', 'cnnData': '결합재비 59.6 산골재율 51.2 0/0 ', 'colLbl': '768'}
    {'location': '742,2135,82,32', 'text': '59.6', 'cnnData': '59.6 산골재율 51.2 0/0 ', 'colLbl': '768'}
    {'location': '1070,2135,197,36', 'text': '산 골 재 율', 'cnnData': '산골재율 51.2 0/0 출하계확인', 'colLbl': '768'}
    {'location': '1482,2135,82,33', 'text': '51.2', 'cnnData': '51.2 0/0 ', 'colLbl': '768'}
    {'location': '1790,2140,26,27', 'text': '0/0', 'cnnData': '0/0 ', 'colLbl': '768'}
    {'location': '1340,2218,121,67', 'text': '7%한', 'cnnData': '7%한 ', 'colLbl': '768'}
    {'location': '269,2225,227,37', 'text': '지 정 사 항', 'cnnData': '지정사항 7%한 인수자확인', 'colLbl': '768'}
    {'location': '1453,2311,260,40', 'text': '4.5 士 15%', 'cnnData': '4.5士15% ', 'colLbl': '768'}
    {'location': '544,2314,56,42', 'text': '염화', 'cnnData': '염화 4.5士15% 물 량:0.3kg/m3이하 공기랑• (성유보강제)상일동역사거리직진3-7게이트329동', 'colLbl': '768'}
    {'location': '600,2314,28,42', 'text': '물', 'cnnData': '물 4.5士15% 량:0.3kg/m3이하 공기랑• ', 'colLbl': '776'}
    {'location': '628,2314,336,42', 'text': '량:0.3kg/m3이하', 'cnnData': '량:0.3kg/m3이하 4.5士15% 공기랑• ', 'colLbl': '768'}
    {'location': '1233,2317,124,37', 'text': '공기랑•', 'cnnData': '공기랑• 4.5士15% ', 'colLbl': '768'}
    {'location': '460,2320,33,26', 'text': '고', 'cnnData': '고 4.5士15% 염화 물 량:0.3kg/m3이하 타', 'colLbl': '768'}
    {'location': '1067,2386,198,36', 'text': '출하계 확인', 'cnnData': '출하계확인 표시사항확인', 'colLbl': '768'}
    {'location': '1442,2406,120,36', 'text': '인 성훈', 'cnnData': '인성훈 ', 'colLbl': '768'}
    {'location': '274,2407,213,36', 'text': '인수자 확인', 'cnnData': '인수자확인 인성훈 ', 'colLbl': '768'}
    {'location': '1058,2430,216,36', 'text': '표시사항확인', 'cnnData': '표시사항확인 ', 'colLbl': '768'}
    {'location': '540,2490,993,42', 'text': '(성유보강제)상일동역사거리 직진 3-7게이트 329동', 'cnnData': '(성유보강제)상일동역사거리직진3-7게이트329동 ', 'colLbl': '768'}
    {'location': '462,2498,31,36', 'text': '타', 'cnnData': '타 (성유보강제)상일동역사거리직진3-7게이트329동 ', 'colLbl': '768'}
    {'location': '967,2616,289,47', 'text': '주식회사 산하', 'cnnData': '주식회사산하 ', 'colLbl': '768'}
    {'location': '249,2648,268,28', 'text': 'B5(182mm ※257mm)', 'cnnData': 'B5(182mm※257mm) ', 'colLbl': '768'}
    '''
    # label mapping
    ocrData = evaluateLabel(ocrData)

    # entry mapping
    ocrData = evaluateEntry(ocrData)

    # findLabel CNN
    # ocrData = labelEval.startEval(ocrData)

    # findEntry CNN
    # ocrData = entryEval.startEval(ocrData)

    obj = {}
    obj["fileName"] = item[item.rfind("/")+1:]
    obj["docCategory"] = {"DOCTYPE": docType, "DOCTOPTYPE": docTopType, "DOCSCORE": maxNum}
    obj["data"] = ocrData
    print(obj)
    return obj

def evaluateEntry(ocrData):
    try:
        for colData in ocrData:
            if 'colLbl' in colData and int(colData['colLbl']) > 0:
                print("colLbl:", colData)
                colLoc = colData['location'].split(',')

                # entryCheck
                for entryData in ocrData:
                    entryLoc = entryData['location'].split(',')

                    # 수평 check
                    if locationCheck(colLoc[1], entryLoc[1], 30, -30):
                        if 'entryLbl' not in entryData and 'colLbl' not in entryData:
                            entryData['entyLbl'] = colData['colLbl']
                            break

                    # 수직 check and colLbl 보다 아래 check
                    elif locationCheck(colLoc[0], entryLoc[0], 100, -100) and locationCheck(colLoc[1], entryLoc[1], 50, -300):
                        if 'entryLbl' not in entryData and 'colLbl' not in entryData:
                            entryData['entryLbl'] = colData['colLbl']
                            break
        return ocrData
    except Exception as e:
        print(e)

def evaluateLabel(ocrData):
    try:
        labelDatas = []
        delDatas = []
        trainData = 'C:/projectWork/icrRest/labelTrain/data/kkk.train'
        f = open(trainData, 'r', encoding='utf-8')
        lines = f.readlines()

        for line in lines:
            data = line.split(',')
            data[1] = data[1].replace('\n', '')
            labelDatas.append(data)

        for data in ocrData:
            text = data['text'].replace(' ', '')
            dataLoc = data['location'].split(',')
            insertDatas = []

            for labelData in labelDatas:
                tempStr = text
                # 아래쪽 일치 확인
                for i in range(4):

                    if labelData[0].lower().find(tempStr.lower()) == 0:
                        # 완전일치 확인
                        if labelData[0].lower() == tempStr.lower():
                            data['colLbl'] = labelData[1]

                            for insertData in insertDatas:
                                data['text'] += ' ' + insertData['text']

                            break
                        else:
                            # 아래 문장 합쳐서 tempStr에 저장
                            for bottomData in ocrData:
                                bottomLoc = bottomData['location'].split(',')

                                # 수직 check and 아래 문장 check
                                if locationCheck(dataLoc[0], bottomLoc[0], 100, -100) and locationCheck(dataLoc[1],
                                                                                                        bottomLoc[1], 0,
                                                                                                        -60):
                                    tempStr += bottomData['text'].replace(' ', '')
                                    delDatas.append(bottomData)
                                    insertDatas.append(bottomData)
                    else:
                        break

                # data에 colLbl이 없으면 오른쪽 일치 확인

        for delData in delDatas:
            ocrData.remove(delData)

        return ocrData
    except Exception as e:
        print(e)

# pdf 에서 png 변환 함수
def convertPdfToImage(upload_path, pdf_file):

    try:
        pages = convert_from_path(upload_path + pdf_file, dpi=300, output_folder=None, first_page=None,
                                  last_page=None,
                                  fmt='ppm', thread_count=1, userpw=None, use_cropbox=False, strict=False,
                                  transparent=False)
        pdf_file = pdf_file[:-4]  # 업로드 파일명
        filenames = []
        for page in pages:
            filename = "%s-%d.jpg" % (pdf_file, pages.index(page))
            page.save(upload_path + filename, "JPEG")
            filenames.append(filename)
        return filenames
    except Exception as e:
        print(e)

def imgResize(filename):
    try:
        # FIX_LONG = 3600
        # FIX_SHORT = 2400

        FIX_LONG = 2970
        FIX_SHORT = 2100
        filenames = []

        imgs = cv2.imreadmulti(filename)[1]
        index = 0

        for i in range(0,len(imgs)):

            img = imgs[i]
            height, width = img.shape[:2]
            imagetype = "hori"
            # 배율
            magnify = 1
            if width - height > 0:
                imagetype = "hori"
                if (width / height) > (FIX_LONG / FIX_SHORT):
                    magnify = round((FIX_LONG / width) - 0.005, 2)
                else:
                    magnify = round((FIX_SHORT / height) - 0.005, 2)
            else:
                imagetype = "vert"
                if (height / width) > (FIX_LONG / FIX_SHORT):
                    magnify = round((FIX_LONG / height) - 0.005, 2)
                else:
                    magnify = round((FIX_SHORT / width) - 0.005, 2)

            # 확대, 축소
            img = cv2.resize(img, dsize=(0, 0), fx=magnify, fy=magnify, interpolation=cv2.INTER_LINEAR)
            height, width = img.shape[:2]
            # 여백 생성
            if imagetype == "hori":
                img = cv2.copyMakeBorder(img, 0, FIX_SHORT - height, 0, FIX_LONG - width, cv2.BORDER_CONSTANT,
                                         value=[255, 255, 255])
            else:
                img = cv2.copyMakeBorder(img, 0, FIX_LONG - height, 0, FIX_SHORT - width, cv2.BORDER_CONSTANT,
                                         value=[255, 255, 255])

            ext = findExt(filename)

            if ext.lower() == '.tif':
                name = filename[:filename.rfind(".")]
                name = "%s-%d.jpg" % (name, index)
                cv2.imwrite(name, img)
                filenames.append(name)
                index = index + 1
            else:
                cv2.imwrite(filename, img)
                filenames.append(filename)

        return filenames

    except Exception as ex:
        raise Exception(
            str({'code': 500, 'message': 'imgResize error', 'error': str(ex).replace("'", "").replace('"', '')}))

def findExt(fileName):
    ext = fileName[fileName.rfind("."):]
    return ext

def get_Ocr_Info(filePath):
    headers = {
        # Request headers
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': 'b54aa37a89f943258a782bf900f0f531',
    }

    params = urllib.parse.urlencode({
        # Request parameters
        'language': 'unk',
        'detectOrientation ': 'true',
    })

    try:
        body = open(filePath, 'rb').read()

        conn = http.client.HTTPSConnection('japaneast.api.cognitive.microsoft.com')
        conn.request("POST", "/vision/v2.0/ocr?%s" % params, body, headers)
        response = conn.getresponse()
        data = response.read()
        data = json.loads(data)
        data = ocrParsing(data)
        conn.close()

        return data
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))

def ocrParsing(body):
    data = []
    for i in body["regions"]:
        for j in i["lines"]:
            item = ""
            for k in j["words"]:
                item += k["text"] + " "
            data.append({"location":j["boundingBox"], "text":item[:-1]})
    return data

# y ?? ??
def sortArrLocation(inputArr):
    tempArr = []
    retArr = []
    for item in inputArr:
        tempArr.append((makeindex(item['location']), item))
    tempArr.sort(key=operator.itemgetter(0))
    for tempItem in tempArr:
        retArr.append(tempItem[1])
    return retArr

def makeindex(location):
    if len(location) > 0:
        temparr = location.split(",")
        for i in range(0, 5):
            if (len(temparr[0]) < 5):
                temparr[0] = '0' + temparr[0]
        return int(temparr[1] + temparr[0])
    else:
        return 999999999999

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def findDocType(ocrData):
    try:
        docTopType = 0
        docType = 0
        text = []
        maxNum = 0
        strText = ''

        file = open('docSentence.txt','r', encoding="UTF8")
        sentenceList = []

        for line in file:
            sentence,type,topType = line.strip().split("||")
            dic = {}
            dic["sentence"] = sentence
            dic["docType"] = type
            dic["docTopType"] = topType
            sentenceList.append(dic)
        file.close()

        regExp = "[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]"

        for i, item in enumerate(ocrData):
            text.append(re.sub(regExp, "", item["text"]))
            strText = ",".join(str(x) for x in text)
            if i == 19:
                break

        strText = strText.lower()

        for rows in sentenceList:
            ratio = similar(strText, rows["sentence"])

            if ratio > maxNum:
                maxNum = ratio
                docType = rows["docType"]
                docTopType = rows["docTopType"]

        if maxNum > 0.1:
            return int(docTopType), int(docType), maxNum
        else:
            return docTopType, docType, maxNum

    except Exception as ex:
        raise Exception(str({'code': 500, 'message': 'findDocType error',
                             'error': str(ex).replace("'", "").replace('"', '')}))

def splitLabel(ocrData):
    try:
        sepKeywordList = []

        # sep_keyword ? ??ì¶ì¶
        file = open("splitLabel.txt", "r", encoding="UTF8")
        for line in file:
            sepKeyword = line.strip()
            sepKeywordList.append(sepKeyword)

        for keyWord in sepKeywordList:
            for item in ocrData:
                if item["text"].replace(" ", "").find(keyWord) > -1:

                    item["text"] = item["text"].replace(" ", "")
                    textLen = len(item["text"])
                    location = item["location"].split(",")
                    value = math.ceil(int(location[2]) / textLen)

                    textList = splitText(item["text"], keyWord)
                    ocrData.remove(item)

                    findLen = 0

                    for idx, i in enumerate(textList):
                        dic = {}
                        dicLoc = ""

                        find = item["text"].find(i, findLen)
                        findLen += len(i)
                        width = int(value * find)

                        if idx == 0:
                            dicLoc = location[0] + "," + location[1] + "," + str(int(value * len(i))) + "," + location[3]
                        else:
                            dicLoc = str(int(location[0]) + width) + "," + location[1] + "," + str(
                                int(value * len(i))) + "," + location[3]

                        dic["location"] = dicLoc
                        dic["text"] = i
                        ocrData.append(dic)

        ocrData = sortArrLocation(ocrData)
        return ocrData

    except Exception as ex:
        raise Exception(str({'code':500, 'message':'splitLabel error', 'error':str(ex).replace("'","").replace('"','')}))

def splitText(text, split):
    result = []

    while True:
        find = text.find(split)

        if find == 0:
            result.append(text[0:len(split)])
            text = text[len(split):]
        elif find > 0:
            result.append(text[0:find])
            result.append(text[find:find + len(split)])
            text = text[find + len(split):]

        if find == -1:
            if len(text) > 0:
                result.append(text)
            break

    return result

def locationCheck(loc1, loc2, plus, minus):
    if minus < int(loc1) - int(loc2) < plus:
        return True
    else :
        return False

def bottomCheck(loc1, loc2, num):
   if int(loc1) - int(loc2) < num:
       return True
   else:
       return False

def extractCNNData(inputArr):
    for item in inputArr:
        yData = []
        xData = []
        itemLoc = item["location"].split(",")

        yData.append(item["text"].replace(" ", ""))
        xData.append(item["text"].replace(" ", ""))

        for data in inputArr:
            dataLoc = data["location"].split(",")

            # 아래로 5개 문장 가져오기
            if item != data and bottomCheck(itemLoc[1], dataLoc[1], 2) and locationCheck(itemLoc[0], dataLoc[0], 10, -10) and len(yData) < 5:
                yData.append(data["text"].replace(" ", ""))

            # 오른쪽으로 5개 문장 가져오기
            if item != data and bottomCheck(itemLoc[0], dataLoc[0], 2) and locationCheck(itemLoc[1], dataLoc[1], 100, -100) and len(xData) < 5:
                xData.append(data["text"].replace(" ", ""))

        xText = ""
        yText = ""

        for x in xData:
            xText += x + " "

        for idx, y in enumerate(yData):
            if idx != 0:
                yText += y + " "

        item["cnnData"] = xText + yText[:-1]
        item["cnnData"] = item["cnnData"].replace(",","")\

        if item["cnnData"][-1] == " ":
            item["cnnData"] = item["cnnData"][:-1]

    return inputArr

def compareLabel(inputArr):

    for item in inputArr:
        yData = []
        xData = []
        itemLoc = item["location"].split(",")

        yData.append(item["text"].replace(" ", ""))
        xData.append(item["text"].replace(" ", ""))

        for data in inputArr:
            dataLoc = data["location"].split(",")

            # 아래로 5개 문장 가져오기
            if item != data and bottomCheck(itemLoc[1], dataLoc[1], 2) and locationCheck(itemLoc[0], dataLoc[0], 10, -10) and len(yData) < 5:
                yData.append(data["text"].replace(" ", ""))

            # 오른쪽으로 5개 문장 가져오기
            if item != data and bottomCheck(itemLoc[0], dataLoc[0], 2) and locationCheck(itemLoc[1], dataLoc[1], 10, -10) and len(xData) < 5:
                xData.append(data["text"].replace(" ", ""))

        xText = ""
        yText = ""

        for x in xData:
            xText += x + " "

        for y in yData:
            yText += y + " "

        item["xData"] = xText[:-1]
        item["yData"] = yText[:-1]

    return inputArr

def findLabel(ocrData):
    ocrData = labelEval.startEval(ocrData)
    return ocrData

def findEntry(ocrData):
    ocrData = entryEval.startEval(ocrData)
    return ocrData


def findColByML(ocrData):
    obj = [{'yData': 'aaa1', 'text': 'bbb1', 'xData': 'ccc1', 'location': 44},
           {'yData': 'aaa2', 'text': 'bbb2', 'xData': 'ccc2', 'location': 530},
           {'yData': 'aaa3', 'text': 'bbb3', 'xData': 'ccc3', 'location': 81},
           {'yData': 'aaa4', 'text': 'bbb4', 'xData': 'ccc4', 'location': 1234},
           {'yData': 'aaa5', 'text': 'bbb5', 'xData': 'ccc5', 'location': 1039}]

    resultObj = {}
    colName = ["xData", "yData", "text", "location"]
    dataArr = []
    for qq in obj:
        tmpArr = [qq.get('xData'),
                  qq.get('yData'),
                  qq.get('text'),
                  qq.get('location')
                  ]
        dataArr.append(tmpArr)

    resultObj['ColumnNames'] = colName;
    resultObj['Values'] = dataArr;

    data = {
        "Inputs": {
            "input1": resultObj,
        },
        "GlobalParameters": {
        }
    }

    body = str.encode(json.dumps(data))
    api_key = 'Glka58B/GkaysKmq01K/1S7zIhiuAPo1k9l1wq/8Z6NjrQGTMJs4cbMXiV0a2Lr5eVggch1aIDQjUDKaCLpYEA=='
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}
    url = 'https://ussouthcentral.services.azureml.net/workspaces/a2de641a3e3a40d7b85125db08cf4a97/services/9ca98ef979444df8b1fcbecc329c46bd/execute?api-version=2.0&details=true'

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        # print(json.dumps(result.decode("utf8", 'ignore')))
        return json.dumps(result.decode("utf8", 'ignore'))
    except urllib.error.HTTPError as error:
        # print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        # print(error.info())
        # print(json.loads(error.read().decode("utf8", 'ignore')))
        return json.loads(error.read().decode("utf8", 'ignore'))


if __name__=='__main__':
    # app.run(host='0.0.0.0',port=5000,debug=True)
    pyOcr('test')