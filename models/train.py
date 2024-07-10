#%% 라이브러리 로드
import os
import pandas as pd
import numpy as np
import json
import re
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, BertTokenizerFast, BertModel
from torch.utils.data import DataLoader, Dataset
import wandb
# from tqdm.auto import tqdm
from datasets import load_metric
from datetime import datetime
from torch.optim import AdamW
import logging
import warnings
from konlpy.tag import Okt
from soynlp.normalizer import repeat_normalize
import torch.nn.functional as F

#%% 전역 변수 
# W&B 로그인
wandb.login(key = wandb.key)

# 전역 변수 설정
seed = 42

# 학습하며 반복적으로 나오는 해결 불가능한 경고 메시지를 무시
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# 오늘 날짜- log 저장 시 사용
today_date = datetime.now().strftime('%m%d%h')

#%% 함수 정의 - 코드 완성되면 다른 폴더로 옮기고 import 하는 방식으로 변경

# 기본 전처리 함수
def clean(x):
    # emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    x = pattern.sub(' ', x)
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x

# 모델 학습에 필요한 데이터 셋으로 만들어주는 클래스
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 한국어 불용어 리스트
stopwords = set([
    '이', '그', '저', '것', '수', '들', '등', '때', '문제', '뿐', '안', '이랑', '랑', 
    '도', '곳', '걸', '에서', '하지만', '그렇지만', '그러나', '그리고', '따라서', 
    '그러므로', '그러나', '그런데', '때문에', '왜냐하면', '무엇', '어디', '어떤', 
    '어느', '어떻게', '누가', '누구', '어떤', '한', '하다', '있다', '되다', '이다', 
    '로', '로서', '로써', '과', '와', '이다', '입니다', '한다', '할', '위해', 
    '또한', '및', '이외', '더불어', '그리고', '따라', '따라서', '뿐만아니라', '그럼', 
    '하지만', '있어서', '그래서', '그렇다면', '이에', '때문에', '무엇', '어디', 
    '어떻게', '왜', '어느', '하는', '하게', '해서', '이러한', '이렇게', '그러한', 
    '그렇게', '저러한', '저렇게', '하기', '한것', '한것이', '일때', '있는', '있는것', 
    '있는지', '여기', '저기', '거기', '뭐', '왜', '어디', '어느', '어떻게', '무엇을', 
    '어디서', '어디에', '무엇인가', '무엇이', '어떤', '누가', '누구', '무엇', 
    '어디', '어떤', '한', '하다', '있다', '되다', '이다', '로', '로서', '로써', 
    '과', '와', '이', '그', '저', '것', '수', '들', '등', '때', '문제', '뿐', 
    '안', '이랑', '랑', '도', '곳', '걸', '에서', '하지만', '그렇지만', '그러나', 
    '그리고', '따라서', '그러므로', '그러나', '그런데', '때문에', '왜냐하면'
])

# KoNLPy Okt 형태소 분석기 로드
okt = Okt()

def remove_stopwords(texts, stopwords, okt):
    """
    입력 리스트에서 불용어를 제거하고 형태소 분석하여 반환하는 함수

    :param texts: 리스트 형식의 텍스트 데이터
    :param stopwords: 불용어 리스트
    :param okt: KoNLPy Okt 형태소 분석기
    :return: 불용어가 제거된 텍스트 리스트
    """
    result = []
    for text in texts:
        tokens = okt.morphs(text)
        filtered_tokens = [token for token in tokens if token not in stopwords]
        result.append(' '.join(filtered_tokens))
    return result

#%% 데이터 불러오기
# train셋
train_data_path = "../data/train.csv"
train_data = pd.read_csv(train_data_path)

label_encode = {
    "협박 대화": 0,
    "갈취 대화": 1,
    "직장 내 괴롭힘 대화": 2,
    "기타 괴롭힘 대화": 3,
}

intimidation = train_data[train_data['class'] == '협박 대화']
extortion = train_data[train_data['class'] == '갈취 대화']
harassment_workplace = train_data[train_data['class'] == '직장 내 괴롭힘 대화']
harassment_others = train_data[train_data['class'] == '기타 괴롭힘 대화']

# 통합 및 라벨링
concat_tr_df = pd.concat([intimidation, extortion, harassment_workplace, harassment_others], axis=0, ignore_index=True)
concat_tr_df['encoded_label'] = concat_tr_df['class'].map(label_encode)

# test 셋
with open('../data/test.json') as f:
    test_data = json.load(f)

test_data = pd.DataFrame(test_data).T
test_data.reset_index(drop=True, inplace=True)

# 실험 셋 로드
exp_set = pd.read_csv('./data/exp_set.csv')

#%% 모델 학습
tmpDf = pd.DataFrame()

for idx, row in exp_set.iterrows():
    train_texts = concat_tr_df['conversation'].to_list()
    train_labels = concat_tr_df['encoded_label'].to_list()
    test_texts = test_data['text'].tolist()

    # 실험 셋에서 하이퍼파라미터 및 전처리 함수 로드
    pre = [None, None, None]  # pre01, pre02, pre03에 대한 함수를 저장할 리스트 초기화
    for i in range(3):  # pre01~n 반복
        pre_key = f'pre0{i+1}'  # 현재 전처리 함수 키 생성
        pre_func_name = row.get(pre_key, None)  # 공백이면 None 반환
        if pre_func_name and pre_func_name in globals():  # 함수 이름이 globals에 존재하는지 확인
            pre[i] = globals()[pre_func_name]  # 함수 객체를 리스트에 저장
        else:
            pre[i] = None  # 함수 이름이 없거나 globals에 없는 경우 None 할당
    test_size = row['test_size']
    model_name = row['model_name']
    learning_rate = row['learning_rate']
    epochs = row['epochs']
    batch_size = row['batch_size']

    # 모델 구성 및 하이퍼파라미터 로깅
    wandb.config = {
        'pre01': pre[0],
        'pre02': pre[1],
        'pre02': pre[2],
        'test_size': test_size,
        'model_name': model_name,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size
    }
    
    # 전처리 목록에 clean 함수가 있으면 적용
    preprocess_func = next((func for func in pre if func is not None and func.__name__ == 'clean'), None)
    if preprocess_func:
        print('전처리 함수 적용')
        train_texts = list(map(clean, train_texts))
        test_texts = list(map(clean, test_texts))
        clean_YN = 'Y'
    else:
        clean_YN = 'N'
        
    # 전처리 목록에 remove_stopwords 함수가 있으면 적용
    preprocess_func = next((func for func in pre if func is not None and func.__name__ == 'remove_stopwords'), None)
    if preprocess_func:
        print('불용어 제거')
        train_texts = preprocess_func(train_texts, stopwords, okt)
        test_texts = preprocess_func(test_texts, stopwords, okt)
        stopwords_YN = 'Y'
    else:
        stopwords_YN = 'N'
    
    # wandb에 기록될 실험 이름 
    exp_name = f"{today_date}_{model_name.replace('/', '_').replace('-', '_')}_{clean_YN}_{stopwords_YN}_{epochs}_{batch_size}"
    
    # 데이터 분할
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=test_size, random_state=seed, stratify=train_labels
    )

    # 토크나이저 및 데이터셋 생성
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # 모델 생성 및 학습
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encode), ignore_mismatched_sizes=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # wandb 초기화
    wandb.init(project="text-multi-label-classification", entity="dogcat1943", name=f"{exp_name}", config=wandb.config)

    # W&B에 모델을 로깅할 수 있도록 W&B에 등록
    wandb.watch(model, log="all")

    # early stopping을 위한 변수 초기화
    best_val_loss = np.inf
    patience = 5  # 성능이 개선되지 않는 에포크의 최대 허용 횟수
    no_improve_epoch = 0
    
    # 학습 루프
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0
        train_metric = load_metric("accuracy", trust_remote_code=True)
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            train_metric.add_batch(predictions=predictions, references=batch["labels"])
        
        train_loss = train_loss / len(train_dataloader)
        train_accuracy = train_metric.compute()["accuracy"]

        # 검증 루프
        model.eval()
        val_loss = 0
        val_steps = 0
        # metric = load_metric("f1", trust_remote_code=True)
        val_metric = load_metric("accuracy", trust_remote_code=True)
        f1_metric = load_metric("f1", trust_remote_code=True)
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            
            val_loss += outputs.loss.item()
            val_steps += 1

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            val_metric.add_batch(predictions=predictions, references=batch["labels"])
            f1_metric.add_batch(predictions=predictions, references=batch["labels"])

        val_loss = val_loss / val_steps
        val_accuracy = val_metric.compute()["accuracy"]
        final_score = f1_metric.compute(average='weighted')

        # 로그 기록
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1_score": final_score["f1"]
        })
        
        # early stopping
        # 성능 개선 확인
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epoch = 0
            # 모델 저장 로직 추가 가능
            torch.save(model.state_dict(), f'./torch_models/{exp_name}.pth')
        else:
            no_improve_epoch += 1

        # 조기 중단 조건 확인
        if no_improve_epoch > patience:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break
    
    # 실험 결과를 DataFrame으로 변환하여 저장
    tmp = exp_set.iloc[idx]
    results = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_f1_score": final_score["f1"]
    }
    for col, value in results.items():
        tmp[col] = value
    tmp2 = pd.DataFrame([tmp])
    tmpDf = pd.concat([tmpDf, tmp2], ignore_index=True)

    # wandb 종료
    wandb.finish()
    
    # 모델 예측
    # 입력 데이터를 디바이스로 이동
    test_encodings = {key: val.to(device) for key, val in test_encodings.items()}

    # 모델을 평가 모드로 전환
    model.eval()

    # 예측 수행
    with torch.no_grad():
        outputs = model(**test_encodings)

    # 로짓(logits)에서 예측 확률 계산
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # 확률 파일 저장 - 추후 앙상블에 사용
    predictions_np = predictions.cpu().numpy()
    predictions_df = pd.DataFrame(predictions_np, columns=['갈취 대화','기타 괴롭힘 대화','직장 내 괴롭힘 대화','협박 대화'])
    # predictions_df['max_val'] = predictions_df.max(axis=1)
    predictions_df['label'] = predictions_df.iloc[:, :4].apply(lambda row: row.argmax(), axis=1)
    predictions_df['class'] = predictions_df.idxmax(axis=1)

    predictions_df.to_csv(f'./test_pred_logit/{exp_name}.csv', index=False, encoding='utf-8-sig')

    # 예측 결과 확인
    predicted_classes = torch.argmax(predictions, dim=1)

    # 텐서를 CPU로 이동시키고 NumPy 배열로 변환
    predicted_classes_np = predicted_classes.cpu().numpy()
    len(predicted_classes_np)

    submission = pd.read_csv('../data/submission.csv')
    submission['class'] = predicted_classes_np

    # 제출 파일 저장
    submission.to_csv(f'./test_pred_clss/{exp_name}.csv', index=False)

# DataFrame을 CSV 파일로 저장
result_csv = pd.read_csv('./data/results.csv')
result_csv = pd.concat([result_csv, tmpDf], ignore_index=True)
result_csv.to_csv('./data/results.csv', index=False)





