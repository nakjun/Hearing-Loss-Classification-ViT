import os
import shutil
import random

def split_dataset(source_dir, train_dir, test_dir, train_ratio=0.8):
    # 카테고리 폴더 가져오기
    categories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for category in categories:
        category_path = os.path.join(source_dir, category)
        images = os.listdir(category_path)
        random.shuffle(images)

        # 데이터 분할
        train_size = int(len(images) * train_ratio)
        train_images = images[:train_size]
        test_images = images[train_size:]

        # 학습 데이터 복사
        train_category_path = os.path.join(train_dir, category)
        os.makedirs(train_category_path, exist_ok=True)
        for image in train_images:
            shutil.move(os.path.join(category_path, image), os.path.join(train_category_path, image))

        # 테스트 데이터 복사
        test_category_path = os.path.join(test_dir, category)
        os.makedirs(test_category_path, exist_ok=True)
        for image in test_images:
            shutil.move(os.path.join(category_path, image), os.path.join(test_category_path, image))

# 경로 설정
source_directory = './dataset/image'
train_directory = './dataset/train'
test_directory = './dataset/test'

# 데이터셋 분할 실행
split_dataset(source_directory, train_directory, test_directory)
