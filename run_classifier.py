import argparse, sys, os
import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from torch.autograd import Variable
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from models.LSTM import LSTMClassifier
import numpy as np
import pickle

def main(args):
    batch_size = 32
    output_size = 2
    hidden_size = 256
    embedding_length = 300

    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=50)
    LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    train_data = data.TabularDataset(path=args.train_data_tsv_file, format='tsv',fields=[('text', TEXT),('label', LABEL)], skip_header=True)
    TEXT.build_vocab(train_data, vectors=GloVe('840B', 300))
    LABEL.build_vocab(train_data)
    word_embeddings = TEXT.vocab.vectors
    vocab_size = len(TEXT.vocab)
    
    model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    model.load_state_dict(torch.load(args.saved_model_path))
    model.cuda()
    model.eval()
    for segments_pkl in os.listdir(args.transcript_segments_folder):
        print(segments_pkl)
        all_segments = pickle.load(open(os.path.join(args.transcript_segments_folder, segments_pkl), 'rb'))
        readable_output_file = open(os.path.join(args.output_transcript_segments_folder, os.path.splitext(segments_pkl)[0]+'.tsv'), 'w')
        for video_id, segments in all_segments.items():
            for i in range(len(segments)):
                sentence = word_tokenize(segments[i]['transcript'].lower())
                test_sent = [[TEXT.vocab.stoi[x] for x in sentence]]
                test_sent = np.asarray(test_sent)
                test_sent = torch.LongTensor(test_sent)
                test_tensor =  Variable(test_sent, volatile=True).cuda()
                output = model(test_tensor, 1)
                out = F.softmax(output, 1)
                if (torch.argmax(out[0]) == 1):
                    pred_label = 0
                else:
                    pred_label = 1
                segments[i]['is_background'] = pred_label 
                all_segments[video_id][i] = segments[i]
                readable_output_file.write('%s\t%d\n' % (' '.join(sentence), pred_label))
        pickle.dump(all_segments, open(os.path.join(args.output_transcript_segments_folder, segments_pkl), 'wb'))
         
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train_data_tsv_file", type = str)
    argparser.add_argument("--saved_model_path", type = str)
    argparser.add_argument("--transcript_segments_folder", type = str)
    argparser.add_argument("--output_transcript_segments_folder", type = str)
    args = argparser.parse_args()
    print (args)
    print ("")
    main(args)

