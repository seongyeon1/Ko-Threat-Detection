import os
import json
import csv

def extract_form_values(directory_path):
    form_values = []

    # 디렉토리 내의 파일 목록을 가져옵니다.
    files = os.listdir(directory_path)
    json_files = [file for file in files if file.endswith('.json')]

    for json_file in json_files:
        file_path = os.path.join(directory_path, json_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 'form' 값을 추출합니다.
            form_list = []
            for document in data.get('document', []):
                for utterance in document.get('utterance', []):
                    form_value = utterance.get('form', '')
                    if form_value:
                        form_list.append(form_value)
            # 'form' 값을 하나의 문자열로 연결합니다.
            form_str = ' '.join(form_list)
            form_values.append([json_file, form_str])

    return form_values

def save_to_csv(data, csv_file_path):
    with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Form Values'])
        writer.writerows(data)

# 사용 예시
directory_path = # 데이터 경로 : 예) "/content/drive/MyDrive/아이펠/mainQ/N_DATA/NIKL_DIALOGUE_2022_v1.0"
csv_file_path =  # 저장 파일경로 : "/content/sample_data/mmc_nomal_data.csv"

form_values = extract_form_values(directory_path)
save_to_csv(form_values, csv_file_path)

print(f"CSV 파일이 {csv_file_path}에 저장되었습니다.")