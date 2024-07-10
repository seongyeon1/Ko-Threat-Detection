from konlpy.tag import Okt
from soynlp.normalizer import repeat_normalize
import re


# %% 함수 정의

# 기본 전처리 함수
def clean(x):
    pattern = re.compile(r'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

    # 영어 알파벳 대소문자와 숫자 제거 정규표현식 추가
    english_pattern = re.compile(r'[a-zA-Z0-9]+')

    x = pattern.sub(' ', x)
    x = url_pattern.sub('', x)

    # 영어 알파벳 대소문자와 숫자 제거
    x = english_pattern.sub('', x)

    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)

    return x


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
    '그리고', '따라서', '그러므로', '그러나', '그런데', '때문에', '왜냐하면', '키키', 'None '
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