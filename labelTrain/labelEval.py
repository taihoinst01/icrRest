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

    #ocrData = json.loads('[{"location":"1018,240,411,87","text":"APEX"},{"location":"1019,338,409,23","text":"Partner of Choice"},{"location":"1562,509,178,25","text":"Voucher No"},{"location":"1562,578,206,25","text":"Voucher Date"},{"location":"206,691,274,27","text":"4153 Korean Re"},{"location":"208,756,525,34","text":"Proportional Treaty Statement"},{"location":"1842,506,344,25","text":"BV/HEO/2018/05/0626"},{"location":"1840,575,169,25","text":"01105/2018"},{"location":"206,848,111,24","text":"Cedant"},{"location":"206,908,285,24","text":"Class of Business"},{"location":"210,963,272,26","text":"Period of Quarter"},{"location":"207,1017,252,31","text":"Period of Treaty"},{"location":"206,1066,227,24","text":"Our Reference"},{"location":"226,1174,145,31","text":"Currency"},{"location":"227,1243,139,24","text":"Premium"},{"location":"226,1303,197,24","text":"Commission"},{"location":"226,1366,107,24","text":"Claims"},{"location":"227,1426,126,24","text":"Reserve"},{"location":"227,1489,123,24","text":"Release"},{"location":"227,1549,117,24","text":"Interest"},{"location":"227,1609,161,31","text":"Brokerage"},{"location":"233,1678,134,24","text":"Portfolio"},{"location":"227,1781,124,24","text":"Balance"},{"location":"574,847,492,32","text":": Solidarity- First Insurance 2018"},{"location":"574,907,568,32","text":": Marine Cargo Surplus 2018 - Inward"},{"location":"598,959,433,25","text":"01-01-2018 TO 31-03-2018"},{"location":"574,1010,454,25","text":": 01-01-2018 TO 31-12-2018"},{"location":"574,1065,304,25","text":": APEX/BORD/2727"},{"location":"629,1173,171,25","text":"JOD 1.00"},{"location":"639,1239,83,25","text":"25.53"},{"location":"639,1299,64,25","text":"5.74"},{"location":"639,1362,64,25","text":"0.00"},{"location":"639,1422,64,25","text":"7.66"},{"location":"639,1485,64,25","text":"0.00"},{"location":"639,1545,64,25","text":"0.00"},{"location":"639,1605,64,25","text":"0.64"},{"location":"648,1677,64,25","text":"0.00"},{"location":"641,1774,81,25","text":"11 .49"},{"location":"1706,1908,356,29","text":"APEX INSURANCE"}]')
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
            if i[0].lower() == row['text'].lower():
                row['colLbl'] = i[1]

    return ocrData

    # with open(out_path, 'w') as f:
    #     csv.writer(f).writerows(predictions_human_readable)

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


#if __name__ == '__main__':
#    startEval('test')