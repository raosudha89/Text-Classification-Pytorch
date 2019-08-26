import argparse
import csv
import pickle
from nltk.tokenize import word_tokenize
import sys
from nltk.corpus import stopwords
from collections import defaultdict

#stopwords = ['is', 'are', 'be', 'does', 'do', 'have', '\'s', 'does', 'was', 'you', 'has', 'come', 'use', 'let', 'make', 'get', 'beef', 'chicken', 'give']
freq_ingredients = ['chicken', 'beef', 'dried', 'fried', 'make', 'mashed', 'get', 'cooking', 'let', 'place', 'sauce']
stopWords = set(stopwords.words('english'))

def main(args):
    verbs_data = pickle.load(open(args.verbs_pkl, 'rb'))
    train_verbs_data = verbs_data['our_training']
    cooking_verb_dict = defaultdict(int)
    for video_id, value in train_verbs_data.items():
         for verb_list in value['verbs']:
            for verb in verb_list:
                if verb not in freq_ingredients and verb not in stopWords:
                    cooking_verb_dict[verb] += 1

    data = pickle.load(open(args.segments_pkl, 'rb'))
    val_data = data['validation']
    with open(args.train_tsv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['text', 'label'])
        for seg_id, segs in val_data.items():
            #print(seg_id)
            sentences = []
            labels = []
            for seg in segs:
                sentence = ' '.join(word_tokenize(seg['transcript'].lower()))
                if seg['is_annotated'] == True:
                    label = 1
                elif seg['is_annotated'] == False:
                    label = 0
                else:
                    print('error')
                    sys.exit(0)
                sentences.append(sentence)
                labels.append(label)
            in_intro = True
            for i in range(len(labels)):
                if in_intro and labels[i] == 1:
                    in_intro = False
                if not in_intro:
                    if labels[i] == 0 and labels[i-1] == 1:
                        if sum(labels[i:i+5]) == 0:
                            continue
                        else:
                            for j in range(5):
                                if labels[i+j] == 1:
                                    break
                                labels[i+j] = 1
                            
            for i in range(len(sentences)):
                if labels[i] == 0:
                    for w in sentences[i].split():
                        if w in cooking_verb_dict and cooking_verb_dict[w] >= 10:
                            #writer.writerow([w.upper()])
                            labels[i] = 1
                            break
                writer.writerow([sentences[i], labels[i]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments_pkl", type=str)
    parser.add_argument("--verbs_pkl", type=str)
    parser.add_argument("--train_tsv_file", type=str)
    args = parser.parse_args()
    main(args)
