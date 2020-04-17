import os
import csv


def get_label(label_csv):
    labels = []
    with open(label_csv) as file:
        for label in csv.reader(file):
            labels.append(label[0])
    return sorted(labels)


def create_label_folder(labels, directory):
    for label in labels:
        path = os.path.join(directory, label)
        os.mkdir(path)


def get_dataset(labels, train_csv, max_item_per_label):
    counter = [0 for _ in labels]
    dataset = []
    with open(train_csv) as file:
        for row in csv.reader(file):
            id, label = row[0].split(';')
            idx_label = labels.index(label)
            if counter[idx_label] >= max_item_per_label:
                continue
            dataset.append((id, label))
            counter[idx_label] += 1
    return dataset


def build_video_from_dataset(dataset, src_dir, dst_dir):
    for item in dataset:
        src_item_path = os.path.join(src_dir, item[0])
        dst_label_path = os.path.join(dst_dir, item[1])
        dst_item_path = os.path.join(dst_label_path, item[0])
        build_video_from_frame(src_item_path, dst_item_path)


def build_video_from_frame(frames_path, video_path):
    os.system(
        f"ffmpeg -framerate 25 -pattern_type glob -i '{frames_path}/*.jpg' '{video_path}.mp4' -y")


if __name__ == "__main__":
    jester_dir = "/home/ds/Data/academic/ta/jester"
    jester_data = os.path.join(jester_dir, "20bn-jester-v1")
    jester_label = os.path.join(jester_dir, "jester-v1-labels.csv")
    jester_train = os.path.join(jester_dir, "jester-v1-train.csv")

    dataset_v3_dir = "/home/ds/Data/academic/dataset_v3"

    labels = get_label(jester_label)
    dataset = get_dataset(labels, jester_train, 100)
    build_video_from_dataset(dataset, jester_data, dataset_v3_dir)
    # print(len(dataset))
    # create_label_folder(labels, dataset_v3_dir)
