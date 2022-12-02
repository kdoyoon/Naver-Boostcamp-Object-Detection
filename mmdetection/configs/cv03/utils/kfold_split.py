import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


json_data = '/opt/ml/dataset/train.json'

with open(json_data) as f: data = json.load(f)

var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
X = np.ones((len(data['annotations']),1))
y = np.array([v[1] for v in var])
groups = np.array([v[0] for v in var])

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=3)

n_fold = 0

for train_idx, val_idx in cv.split(X, y, groups):

    print("fold number:", n_fold)
    train_img_fold = []
    train_ann_fold = []
    val_img_fold = []
    val_ann_fold = []

    # TODO : 한 이미지에 대해 중복되는 어노테이션 처리
    """
    이미지 파일 하나에 대해 annotations 딕셔너리를 탐색하며 image_id가 같은 경우
    """

    for gidx in range( len(list(set(groups[train_idx])))-1):

        t = list(set(groups[train_idx]))[gidx]

        for i in range(t,len(data['annotations'])):
            if data['annotations'][i]['image_id'] == t:              
                train_ann_fold.append(data['annotations'][i]) 
            elif data['annotations'][i]['image_id'] > t:
                break



     # 하나의 fold에 대한 인덱스에 따라 이미지 저장, 단 어노테이션과 달리 이미지는 중복이 허용되지 않으므로 중복을 제거
    for gidx in list(set(groups[train_idx])):
        train_img_fold.append(data['images'][gidx])

    for gidx in range(len(list(set(groups[val_idx])))-1):

        t = list(set(groups[val_idx]))[gidx]

        for i in range(t,len(data['annotations'])):
            if data['annotations'][i]['image_id'] == t:              
                val_ann_fold.append(data['annotations'][i]) 
            elif data['annotations'][i]['image_id'] > t:
                break

    for gidx in list(set(groups[val_idx])):
        val_img_fold.append(data['images'][gidx])


    # TODO : 분리 저장한 리스트를 다시 json으로 저장 -> 재효님 split 코드 차용

    t_keys = list(data.keys())
    t_values = [data['info'], data['licenses'], train_img_fold, data['categories'], train_ann_fold]
    t_dict = dict(zip(t_keys, t_values))


    with open(f'/opt/ml/dataset/custom3/train{n_fold}.json', 'w') as f : 
        json.dump(t_dict, f, indent=4)

    v_keys = list(data.keys())
    v_values = [data['info'], data['licenses'], val_img_fold, data['categories'], val_ann_fold]
    v_dict = dict(zip(v_keys, v_values))

    with open(f'/opt/ml/dataset/custom3/valid{n_fold}.json', 'w') as f : 
        json.dump(v_dict, f, indent=4)

    # 폴드 번호 증가
    n_fold += 1

print("Done StratifiedGroupKFold!")