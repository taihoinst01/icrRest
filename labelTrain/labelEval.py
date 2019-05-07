#! /usr/bin/env python
# -*- coding: utf-8 -*-
import json

import tensorflow as tf
import numpy as np
import os
import data_helpers
from multi_class_data_loader import MultiClassDataLoader
from word_data_processor import WordDataProcessor
import csv
import sys

def startEval(ocrData):
    # Parameters
    # ==================================================
    del_all_flags(tf.flags.FLAGS)

    # ocrData = json.loads('[{"location": "1853,269,84,54", "text": "曇.", "cnnData": "曇."}, {"location": "267,377,199,66", "text": "1-174", "cnnData": "1-174 시멘트 물 지정사항 인수자확인"}, {"location": "643,453,591,66", "text": "레디 리스트 콘크리트", "cnnData": "레디리스트콘크리트 납품서 잔골재"}, {"location": "1274,455,176,64", "text": "납품서", "cnnData": "납품서 대표이사전찬7 경기도남양주시와부옵수레로 슬럼프또는 슬럼프플로"}, {"location": "1312,554,163,63", "text": "4體.", "cnnData": "4體."}, {"location": "280,649,40,34", "text": "0", "cnnData": "0 표준명 :레디믹스트콘크리트 132-81-13908 0표준번호.KSF4009 0인증번호.제96-03-026호 0인증기관:한국표준협회 0인|증종류:보통포장고강도콘크리트"}, {"location": "320,649,120,34", "text": "표준명", "cnnData": "표준명 :레디믹스트콘크리트 132-81-13908 결합재비"}, {"location": "440,649,400,34", "text": ":레디믹스트콘크리트", "cnnData": ":레디믹스트콘크리트 132-81-13908"}, {"location": "1342,652,379,32", "text": "132-81-13908", "cnnData": "132-81-13908 주식회사é!0 7%한"}, {"location": "1090,667,144,36", "text": "등록번호", "cnnData": "등록번호 대표전화"}, {"location": "280,690,410,32", "text": "0표 준 번 호 . KS F 4009", "cnnData": "0표준번호.KSF4009 0인증번호.제96-03-026호 0인증기관:한국표준협회 0인|증종류:보통포장고강도콘크리트 대림산업(주)"}, {"location": "1027,720,207,48", "text": "亐 상 호", "cnnData": "亐상호 주식회사é!0 •貳&譬4 급성명 자"}, {"location": "1341,722,364,51", "text": "주식회사 é!0", "cnnData": "주식회사é!0 •貳&譬4 7%한"}, {"location": "1683,729,179,75", "text": "•貳&譬4", "cnnData": "•貳&譬4"}, {"location": "280,731,480,32", "text": "0인 증 번 호 . 제96-03-026호", "cnnData": "0인증번호.제96-03-026호 주식회사é!0 •貳&譬4 0인증기관:한국표준협회 0인|증종류:보통포장고강도콘크리트 대림산업(주) No"}, {"location": "280,772,431,33", "text": "0인 증 기 관 : 한국표준협회", "cnnData": "0인증기관:한국표준협회 0인|증종류:보통포장고강도콘크리트 대림산업(주) No 납품장소고덕대림아파트현장"}, {"location": "1274,803,406,44", "text": "대표이사 전 찬 7", "cnnData": "대표이사전찬7 경기도남양주시와부옵수레로 슬럼프또는 슬럼프플로 0㉣㉭㉭"}, {"location": "1026,810,205,39", "text": "급 성 명", "cnnData": "급성명 대표이사전찬7 자"}, {"location": "280,813,643,35", "text": "0인| 증 종 류 : 보통,포장,고강도 콘크리트", "cnnData": "0인|증종류:보통포장고강도콘크리트 급성명 대림산업(주) No 납품장소고덕대림아파트현장 운반차번호"}, {"location": "350,883,496,42", "text": "20 18년 11 월 08 일", "cnnData": "2018년11월08일 경기도남양주시와부옵수레로 소 시멘트"}, {"location": "1273,885,475,39", "text": "경기도 남양주시 와부옵 수레로", "cnnData": "경기도남양주시와부옵수레로 슬럼프또는 슬럼프플로 0㉣㉭㉭"}, {"location": "1199,890,34,30", "text": "소", "cnnData": "소 경기도남양주시와부옵수레로"}, {"location": "1026,916,35,36", "text": "자", "cnnData": "자"}, {"location": "880,940,74,41", "text": "귀하", "cnnData": "귀하 의최대 시방배합표(kg/m3)"}, {"location": "288,941,287,48", "text": "대림산업(주)", "cnnData": "대림산업(주) 귀하 No 납품장소고덕대림아파트현장 운반차번호 납품시각"}, {"location": "1090,965,146,35", "text": "대표전화", "cnnData": "대표전화 031-576-4545출하실031-576-3131"}, {"location": "1262,968,557,36", "text": "031-576-4545 출하실 031-576-3131", "cnnData": "031-576-4545출하실031-576-3131"}, {"location": "403,1051,74,40", "text": "013", "cnnData": "013 타설완료•0응시수"}, {"location": "1142,1052,528,71", "text": "타설완료• 0응 시 수", "cnnData": "타설완료•0응시수"}, {"location": "289,1065,36,25", "text": "No", "cnnData": "No 납품장소고덕대림아파트현장 운반차번호 납품시각 납품용적"}, {"location": "287,1153,694,44", "text": "납 품 장 소 고덕 대림 아파트 현장", "cnnData": "납품장소고덕대림아파트현장 운반차번호 납품시각 납품용적 호칭방법"}, {"location": "663,1248,404,44", "text": "(175) 서울14다7478", "cnnData": "(175)서울14다7478 이은우"}, {"location": "286,1252,194,37", "text": "운반차번호", "cnnData": "운반차번호 (175)서울14다7478 이은우 납품시각 납품용적 호칭방법"}, {"location": "1439,1252,129,39", "text": "이은우", "cnnData": "이은우 인성훈"}, {"location": "858,1342,46,36", "text": "08", "cnnData": "08 03 25mm18••曲"}, {"location": "1247,1342,45,36", "text": "03", "cnnData": "03"}, {"location": "287,1392,194,36", "text": "납 품 시 각", "cnnData": "납품시각 납품용적 호칭방법"}, {"location": "543,1439,133,36", "text": "도 착", "cnnData": "도착 종류에따른구분 보통콘크리트 회수수 염화"}, {"location": "751,1522,91,37", "text": "6.00", "cnnData": "6.00 6.00 계 m3 59.6"}, {"location": "1547,1524,91,36", "text": "6.00", "cnnData": "6.00 m3 시멘트"}, {"location": "287,1525,191,37", "text": "납 품 용 적", "cnnData": "납품용적 6.00 6.00 계 m3 호칭방법"}, {"location": "1185,1529,31,35", "text": "계", "cnnData": "계 6.00 m3 ㉭"}, {"location": "1762,1531,34,35", "text": "m3", "cnnData": "m3 혼화제"}, {"location": "559,1600,180,33", "text": "콘크리트의", "cnnData": "콘크리트의 굵은골재 의최대 슬럼프또는 시멘트 회수수"}, {"location": "739,1600,144,33", "text": "굵은골재", "cnnData": "굵은골재 의최대 슬럼프또는 시멘트 종류에 잔골재 59.6"}, {"location": "883,1600,108,33", "text": "의최대", "cnnData": "의최대 슬럼프또는 시멘트 종류에 시방배합표(kg/m3)"}, {"location": "1277,1603,171,31", "text": "슬럼프 또는", "cnnData": "슬럼프또는 시멘트 종류에 슬럼프플로 0㉣㉭㉭"}, {"location": "1552,1604,102,32", "text": "시멘트", "cnnData": "시멘트 종류에"}, {"location": "1654,1604,102,32", "text": "종류에", "cnnData": "종류에"}, {"location": "1056,1623,129,32", "text": "호칭강도", "cnnData": "호칭강도 굵은골재 표시사항확인"}, {"location": "536,1644,231,33", "text": "종류에따른구분", "cnnData": "종류에따른구분 치수에따른구분 슬럼프플로 따른구분 보통콘크리트 염화 (성유보강제)상일동역사거리직진3-7게이트329동"}, {"location": "767,1644,231,33", "text": "치수에따른구분", "cnnData": "치수에따른구분 슬럼프플로 따른구분"}, {"location": "1276,1646,172,31", "text": "슬럼프 플로", "cnnData": "슬럼프플로 따른구분 0㉣㉭㉭"}, {"location": "1585,1648,137,31", "text": "따른 구분", "cnnData": "따른구분"}, {"location": "284,1662,193,37", "text": "호 칭 방 법", "cnnData": "호칭방법"}, {"location": "1496,1697,140,31", "text": "포를랜드", "cnnData": "포를랜드 시멘트 1종"}, {"location": "1636,1697,105,31", "text": "시멘트", "cnnData": "시멘트 1종"}, {"location": "1741,1697,70,31", "text": "1종", "cnnData": "1종"}, {"location": "535,1711,211,31", "text": "보통콘크리트", "cnnData": "보통콘크리트 25mm18••曲 150mm 염화 (성유보강제)상일동역사거리직진3-7게이트329동"}, {"location": "854,1714,374,37", "text": "25 mm 18 ,••曲", "cnnData": "25mm18••曲 150mm"}, {"location": "1324,1714,146,37", "text": "150 mm", "cnnData": "150mm"}, {"location": "877,1799,323,41", "text": "시방 배합표(kg/m3)", "cnnData": "시방배합표(kg/m3)"}, {"location": "258,1877,93,28", "text": "시멘트", "cnnData": "시멘트 시멘트 회수수 잔골재 잔골재 B5(182mm※257mm)"}, {"location": "351,1877,93,28", "text": "시멘트", "cnnData": "시멘트 회수수 잔골재 잔골재 산골재"}, {"location": "551,1878,93,36", "text": "회수수", "cnnData": "회수수 잔골재 잔골재 산골재 굵은골재 염화"}, {"location": "644,1878,93,36", "text": "잔골재", "cnnData": "잔골재 잔골재 산골재 굵은골재 굵은골재"}, {"location": "737,1878,93,36", "text": "잔골재", "cnnData": "잔골재 산골재 굵은골재 굵은골재 굵은골재 59.6"}, {"location": "830,1878,93,36", "text": "산골재", "cnnData": "산골재 굵은골재 굵은골재 굵은골재 혼화재"}, {"location": "923,1878,124,36", "text": "굵은골재", "cnnData": "굵은골재 굵은골재 굵은골재 혼화재 혼화재"}, {"location": "1047,1878,124,36", "text": "굵은골재", "cnnData": "굵은골재 굵은골재 혼화재 혼화재 혼화재"}, {"location": "1171,1878,124,36", "text": "굵은골재", "cnnData": "굵은골재 혼화재 혼화재 혼화재 혼화제 ㉭"}, {"location": "1295,1878,93,36", "text": "혼화재", "cnnData": "혼화재 혼화재 혼화재 혼화제 혼화제 43"}, {"location": "1388,1878,93,36", "text": "혼화재", "cnnData": "혼화재 혼화재 혼화제 혼화제 혼화제"}, {"location": "1481,1878,93,36", "text": "혼화재", "cnnData": "혼화재 혼화제 혼화제 혼화제 51.2"}, {"location": "1574,1878,93,36", "text": "혼화제", "cnnData": "혼화제 혼화제 혼화제 2.0"}, {"location": "1667,1878,93,36", "text": "혼화제", "cnnData": "혼화제 혼화제"}, {"location": "1760,1878,93,36", "text": "혼화제", "cnnData": "혼화제"}, {"location": "480,1887,26,26", "text": "물", "cnnData": "물 회수수 잔골재 잔골재 산골재"}, {"location": "683,1916,322,26", "text": "0 ㉣ ㉭ 0", "cnnData": "0㉣㉭0 0㉣㉭㉭ ㉭ 467467"}, {"location": "1278,1917,511,25", "text": "0 ㉣ ㉭ ㉭", "cnnData": "0㉣㉭㉭"}, {"location": "1178,1918,23,24", "text": "㉭", "cnnData": "㉭ 0㉣㉭㉭"}, {"location": "691,2017,149,25", "text": "467 467", "cnnData": "467467 91 43 2.0"}, {"location": "992,2017,28,25", "text": "91", "cnnData": "91 43 2.0"}, {"location": "1302,2017,31,25", "text": "43", "cnnData": "43 2.0"}, {"location": "1569,2017,46,25", "text": "2.0", "cnnData": "2.0"}, {"location": "269,2133,45,36", "text": "물", "cnnData": "물 결합재비 59.6 산골재율 51.2 지정사항 인수자확인"}, {"location": "314,2133,180,36", "text": "결합재비", "cnnData": "결합재비 59.6 산골재율 51.2 0/0"}, {"location": "742,2135,82,32", "text": "59.6", "cnnData": "59.6 산골재율 51.2 0/0"}, {"location": "1070,2135,197,36", "text": "산 골 재 율", "cnnData": "산골재율 51.2 0/0 출하계확인"}, {"location": "1482,2135,82,33", "text": "51.2", "cnnData": "51.2 0/0"}, {"location": "1790,2140,26,27", "text": "0/0", "cnnData": "0/0"}, {"location": "1340,2218,121,67", "text": "7%한", "cnnData": "7%한"}, {"location": "269,2225,227,37", "text": "지 정 사 항", "cnnData": "지정사항 7%한 인수자확인"}, {"location": "1453,2311,260,40", "text": "4.5 士 15%", "cnnData": "4.5士15%"}, {"location": "544,2314,56,42", "text": "염화", "cnnData": "염화 4.5士15% 물 량:0.3kg/m3이하 공기랑• (성유보강제)상일동역사거리직진3-7게이트329동"}, {"location": "600,2314,28,42", "text": "물", "cnnData": "물 4.5士15% 량:0.3kg/m3이하 공기랑•"}, {"location": "628,2314,336,42", "text": "량:0.3kg/m3이하", "cnnData": "량:0.3kg/m3이하 4.5士15% 공기랑•"}, {"location": "1233,2317,124,37", "text": "공기랑•", "cnnData": "공기랑• 4.5士15%"}, {"location": "460,2320,33,26", "text": "고", "cnnData": "고 4.5士15% 염화 물 량:0.3kg/m3이하 타"}, {"location": "1067,2386,198,36", "text": "출하계 확인", "cnnData": "출하계확인 표시사항확인"}, {"location": "1442,2406,120,36", "text": "인 성훈", "cnnData": "인성훈"}, {"location": "274,2407,213,36", "text": "인수자 확인", "cnnData": "인수자확인 인성훈"}, {"location": "1058,2430,216,36", "text": "표시사항확인", "cnnData": "표시사항확인"}, {"location": "540,2490,993,42", "text": "(성유보강제)상일동역사거리 직진 3-7게이트 329동", "cnnData": "(성유보강제)상일동역사거리직진3-7게이트329동"}, {"location": "462,2498,31,36", "text": "타", "cnnData": "타 (성유보강제)상일동역사거리직진3-7게이트329동"}, {"location": "967,2616,289,47", "text": "주식회사 산하", "cnnData": "주식회사산하"}, {"location": "249,2648,268,28", "text": "B5(182mm ※257mm)", "cnnData": "B5(182mm※257mm)"}]')
    # Eval Parameters
    print('==============startEval=================')
    print(ocrData)

    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
    tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    data_loader = MultiClassDataLoader(tf.flags, WordDataProcessor())
    data_loader.define_flags()

    FLAGS = tf.flags.FLAGS
    # FLAGS._parse_flags()
    FLAGS(sys.argv)
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    if FLAGS.eval_train:
        x_raw, y_test = data_loader.load_data_and_labels()
        y_test = np.argmax(y_test, axis=1)
    else:
        # x_raw, y_test = data_loader.load_dev_data_and_labels()
        x_raw, y_test = data_loader.load_dev_data_and_labels_json(ocrData)
        y_test = np.argmax(y_test, axis=1)

    # checkpoint_dir이 없다면 가장 최근 dir 추출하여 셋팅
    if FLAGS.checkpoint_dir == "":
        all_subdirs = ["C:/projectWork/icrRest/labelTrain/runs/" + d for d in os.listdir('C:/projectWork/icrRest/labelTrain/runs/.') if os.path.isdir("C:/projectWork/icrRest/labelTrain/runs/" + d)]
        latest_subdir = max(all_subdirs, key=os.path.getmtime)
        FLAGS.checkpoint_dir = latest_subdir + "/checkpoints/"

    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = data_loader.restore_vocab_processor(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    print("\nEvaluating...\n")

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

    # Save the evaluation to a csv
    class_predictions = data_loader.class_labels(all_predictions.astype(int))
    predictions_human_readable = np.column_stack((np.array(x_raw), class_predictions))
    out_path = os.path.join(FLAGS.checkpoint_dir, "../../../", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))

    for i in predictions_human_readable:
        for row in ocrData:
            if i[0].lower() == row['cnnData'].lower():
                row['colLbl'] = i[1]

    # for data in ocrData:
    #     print(data)

    return ocrData

    # with open(out_path, 'w') as f:
    #     csv.writer(f).writerows(predictions_human_readable)

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


# if __name__ == '__main__':
#    startEval('test')