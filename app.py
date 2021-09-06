import numpy as np
import re
import torch
import torch.nn.functional as F
import gluonnlp as nlp
import numpy as np

from torch import nn
from torch.utils.data import Dataset
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from konlpy.tag import Mecab
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

mecab = Mecab("C:/mecab/mecab-ko-dic")


device = torch.device("cuda")

bertmodel, vocab = get_pytorch_kobert_model()

#데이터셋을 bert모델에 들어갈 수 있도록 변환
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

#이진 분류기 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

path = r'C:\Users\hyun5\pub\term_extention'
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

#저장된 모델 불러오기
model = torch.load(r'C:\Users\hyun5\pub\term_extention\model_prototype.pt')
model.eval()

max_len = 64
batch_size = 64

#토크나이저 생성
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def predict_logits(predict_sentence): #모델 예측
    prep_sentence = preprocessing(predict_sentence)
    data = [prep_sentence, '0']
    dataset_another = [data]

    test_dataset = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)
        
        logits = out.detach().cpu()
        logits = torch.sigmoid(logits).numpy()

        return logits

def predict_result(logits):
    if np.argmax(logits) == 0:
        print(logits)
        result = '불공정 약관이 아닙니다.'
    else:
        print(logits)
        result = '{}%의 불공정 약관 가능성이 있습니다.'.format(int(logits[0][1]*100))

    return result


stop_words=['뭐','으면','을','의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','본', '로', '등', '이고', '라', '함']

#특수기호 제거
def clean_text(texts):
    corpus = []
    for i in range(0, len(texts)):
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"\‘\’\“\“\”\○○]', '',str(texts[i])) #remove punctuation
        #review = re.sub(r'\d+','', str(texts[i]))# remove number
        review = review.lower() #lower case
        review = re.sub(r'\s+', ' ', review) #remove extra space
        review = re.sub(r'<[^>]+>','',review) #remove Html tags
        review = re.sub(r'\s+', ' ', review) #remove spaces
        review = re.sub(r"^\s+", '', review) #remove space from start
        review = re.sub(r'\s+$', '', review) #remove space from the end
        corpus.append(review)
    return corpus

#입력 데이터 전처리
def preprocessing(texts):
    cleaned = clean_text(texts)
    tokens = mecab.morphs(''.join(cleaned)) # 토큰화
    tokens = [word for word in tokens if not word in stop_words] # 불용어 제거
    tokens = " ".join(tokens)
    pre_sentence = tokens

    return pre_sentence

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index_web.html')

#web html
@app.route('/web', methods=['POST'])
def web():
    sentence = request.form['sentence']
        
    if (sentence == ""):
         result = "약관을 입력해 주세요!"
        
    else :
        pre_sentence = preprocessing(sentence) #입력 데이터 전처리
        logits = predict_logits(pre_sentence)
        result = predict_result(logits)

    return render_template('index_web.html', result = result)

if __name__ == "__main__": 
    app.run(host="0.0.0.0", port=9000)