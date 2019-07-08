# coding:utf-8
# coding:utf-8
import tensorflow as tf

CHECKPOINT_PATH = "./seq2seq_ckpt-8000"

HIDDEN_SIZE = 1024
NUM_LAYERS = 2
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 4000
BATCH_SIZE = 100
SHARE_EMB_AND_SOFTMAX = True
SOS_ID = 1
EOS_ID = 2




class NMTModel(object):
    def __init__(self):
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) \
                                                     for _ in range(NUM_LAYERS)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) \
                                                     for _ in range(NUM_LAYERS)])

        self.src_embedding = tf.get_variable(
            "src_emb", [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable(
            "trg_emb", [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable("weight", [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable("sotfmax_bias", [TRG_VOCAB_SIZE])

    def inference(self, src_input):
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(
                self.enc_cell, src_emb, src_size, dtype=tf.float32)
        MAX_DEC_LEN = 100

        with tf.variable_scope("decode/rnn/multi_rnn_cell"):
            init_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)
            init_array = init_array.write(0, SOS_ID)

            init_loop_var = (enc_state, init_array, 0)

            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(
                    tf.logical_and(tf.not_equal(trg_ids.read(step), EOS_ID), tf.less(step, MAX_DEC_LEN - 1)))

            def loop_body(state, trg_ids, step):
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

                dec_outputs, next_state = self.dec_cell.call(state=state, inputs=trg_emb)
                output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
                logits = (tf.matmul(output, self.softmax_weight) + self.softmax_bias)
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)

                trg_ids = trg_ids.write(step + 1, next_id[0])
                return next_state, trg_ids, step + 1

            state, trg_ids, step = tf.while_loop(
                continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()


def main(sentence="It is very beautiful!"):
    from stanfordcorenlp import StanfordCoreNLP
    nlp = StanfordCoreNLP("snlp", lang='en')
    with tf.variable_scope("nmt_model", reuse=None):
        model = NMTModel()
    vocab_file = "train.tags.en-zh.en.deletehtml.vocab"
    with open(vocab_file, 'r', encoding='utf-8') as f:
        data = f.readlines()
        words = [w.strip() for w in data]
    word_to_id = {k: v for (k, v) in zip(words, range(len(words)))}
    wordlist = nlp.word_tokenize(sentence.strip()) + ["<eos>"]

    # print(wordlist)
    idlist = [str(word_to_id[w]) if w in word_to_id else str(word_to_id["<unk>"]) for w in wordlist]
    idlist = [int(i) for i in idlist]
    # print(idlist)
    output_op = model.inference(idlist)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7, allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state('./')
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    saver.restore(session, ckpt.model_checkpoint_path)

    vocab_file2 = "train.tags.en-zh.zh.deletehtml.vocab"
    with open(vocab_file2, 'r', encoding='utf-8') as f2:
        data2 = f2.readlines()
        words = [w.strip() for w in data2]
    id_to_word = {k: v for (k, v) in zip(range(len(words)), words)}

    while sentence!='exit':
        wordlist = nlp.word_tokenize(sentence.strip()) + ["<eos>"]
        # print(wordlist)
        idlist = [str(word_to_id[w]) if w in word_to_id else str(word_to_id["<unk>"]) for w in wordlist]
        idlist = [int(i) for i in idlist]
        # print(idlist)
        output_op = model.inference(idlist)
        output = session.run(output_op)
        print([id_to_word[i] for i in output])

        sentence = input('输入你的句子:\n\t').strip()

    session.close()
    nlp.close()


if __name__ == '__main__':

    main()