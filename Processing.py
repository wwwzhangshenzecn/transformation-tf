'''
参考地址：
https://blog.csdn.net/qq_22194315/article/details/85095599
'''

# 此文件是对文本进行预处理
# 采用的粉刺工具是stanfordcorrenlp

import collections
from operator import itemgetter
from stanfordcorenlp import StanfordCoreNLP
import tqdm

def deletehtml(filename1, filename2):
    '''
    将英文和中文弄成一行一句的格式
    :param filename1:
    :param filename2:
    :return:
    '''
    f1 = open(filename1, 'r', encoding='utf-8')
    f2 = open(filename2, 'r', encoding='utf-8')

    data1 = f1.readlines()
    data2 = f2.readlines()

    assert len(data1) == len(data2)

    fw1 = open(filename1 + '.deletehtml', 'w', encoding='utf-8')
    fw2 = open(filename2 + '.deletehtml', 'w', encoding='utf-8')

    print('deletehtml...')

    for line1, line2 in tqdm.tqdm(zip(data1, data2)):
        line1 = line1.strip()
        line2 = line2.strip()
        if line1 and line2:
            if '<' not in line1 and '>' not in line1 and '<' not in line2 and '>' not in line2:
                fw1.write(line1+'\n')
                fw2.write(line2+'\n')

    f1.close()
    f2.close()
    fw1.close()
    fw2.close()

    return filename1 + '.deletehtml', filename2 + '.deletehtml'


def segement_sentence(filename, vovab_size, lang='en'):
    '''
    分词并建立词库
    :param flilename:
    :param vovab_size:
    :param lang:
    :return:
    '''


    nlp = StanfordCoreNLP('snlp', lang=lang)
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
        counter = collections.Counter()
        f1 = open(filename + '.segement', 'w', encoding='utf-8')
        print('Segement {}>>>....'.format(lang))

        for line in tqdm.tqdm(data):
            line = line.strip()
            word_list = nlp.word_tokenize(line)
            setence = ' '.join(word_list)
            try:
                f1.write(setence + '\n')
            except:
                print(setence)
                raise
            for word in word_list:
                counter[word] += 1

        f1.close()

    nlp.close()
    sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = ['<unk>', '<sos>', '<eos>'] + [x[0] for x in sorted_word_to_cnt]

    if len(sorted_words) > vovab_size:
        sorted_words = sorted_words[:vovab_size]

    assert len(sorted_words) <= vovab_size
    with open(filename + '.vocab', 'w', encoding='utf-8') as fw:
        for word in sorted_words:
            fw.write(word + '\n')

    return filename + '.segement'


def convert_to_id(filename, vocab_file):
    '''
    将文本转化为数字编号
    :param filename:
    :param vocab_file:
    :return:
    '''
    with open(vocab_file, 'r', encoding='utf-8') as f:
        data = f.readlines()
        vocab = [w.strip() for w in data]

    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
        f1 = open(filename + '.id', 'w')
        print('Convert ....')

        for line in tqdm.tqdm(data):
            words = line.strip().split() + ['<eos>']
            ids = ' '.join(
                [str(word_to_id[word]) if word in word_to_id else str(word_to_id['<unk>']) for word in words])
            f1.write(ids+'\n')
        f1.close()

    return filename+'.id'


def main():
    '''

    :return:
    '''
    src = 'train.tags.en-zh.en' # 带html标记的文本
    trg = 'train.tags.en-zh.zh'

    src_vocab_size = 10000
    trg_vocab_size = 4000

    #删除html标记，初步处理
    src1, trg1 = deletehtml(src, trg)

    trg2 = segement_sentence(trg1, trg_vocab_size, lang='zh') # 中文分词
    # src2 = segement_sentence(src1, src_vocab_size, lang='en') # 英文分词

    src3 = convert_to_id(src+'.deletehtml.segement', src+'.deletehtml.vocab')
    trg3 = convert_to_id(trg+'.deletehtml.segement', trg+'.deletehtml.vocab')

if __name__ == '__main__':
    main()