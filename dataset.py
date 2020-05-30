from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os
import cv2


class DatasetSplit:
    train = 'train'
    val = 'val'
    test = 'test'


class VideoRecord(object):

    def __init__(self, path, label):
        self.path = path
        self.label = label


class VideoDataset(Dataset):

    def __init__(self, directory, num_segments=16, split=DatasetSplit.train, transform=None):
        self.directory = directory
        self.num_segments = num_segments
        self.split = split
        self.transform = transform
        self.label_set = self._get_label_set()
        self.data = self._get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video = []
        capture = cv2.VideoCapture(self.data[idx].path)
        success, frame = capture.read()
        video.append(frame)
        while success:
            success, frame = capture.read()
            if success:
                video.append(frame)

        frames = self._get_frames(video)
        label_idx = self._get_label_idx(self.data[idx].label)
        if self.transform:
            frames = self._transform_frames(frames)
        return frames, label_idx

    def _get_label_set(self):
        return sorted([x for x in os.listdir(self.directory) if not x.endswith('.py')])

    def _get_label_idx(self, label_name):
        return self.label_set.index(label_name)

    def _transform_frames(self, frames):
        pil_imgs = [[Image.fromarray(f).convert('RGB')]
                    for f in frames]
        img_transforms = [self.transform(f) for f in pil_imgs]
        input_transforms = [f.view(1, 3, f.size(1), f.size(2))
                            for f in img_transforms]
        return input_transforms

    def _get_data(self):
        label_folders = sorted(
            [x for x in os.listdir(self.directory) if not x.endswith('.py')])
        data = []
        for lfold in label_folders:
            lpath = os.path.join(self.directory, lfold)
            video_files = sorted(
                [x for x in os.listdir(lpath) if x.endswith('.mp4')])

            start, end = self._get_split_offset(len(video_files))
            video_files = video_files[start:end]
            for vfile in video_files:
                data.append(VideoRecord(
                    f"{self.directory}/{lfold}/{vfile}", lfold))

        return data

    def _get_split_offset(self, total_length):
        if self.split == DatasetSplit.train:
            return 0, int(0.6 * total_length)
        elif self.split == DatasetSplit.val:
            return int(0.6*total_length), int(0.8*total_length)
        elif self.split == DatasetSplit.test:
            return int(0.8*total_length), int(total_length)
        else:
            # unsupported split
            return 0, 0

    def _get_frames(self, video):
        if len(video) < self.num_segments:
            frames = [x for x in video]
            for i in range(self.num_segments-len(video)):
                frames.append(video[len(video)-1])
            return frames
        else:
            if len(video) < 2*self.num_segments:
                start = int((len(video)-self.num_segments)/2)
                end = start + self.num_segments
                frames = [video[i] for i in range(start, end)]
                return frames
            else:
                step = int(len(video)/self.num_segments)
                i, counter = 0, 0
                frames = []
                while i < len(video) and counter < self.num_segments:
                    frames.append(video[i])
                    i += step
                    counter += 1

                return frames


class VideoLoader():

    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform

    def _get_loader(self, split, batch_size, shuffle, num_workers, pin_memory):
        dataset = VideoDataset(
            self.directory, split=split, transform=self.transform)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory)

    def get_train_loader(self, batch_size=10, shuffle=True, num_workers=8, pin_memory=False):
        return self._get_loader(DatasetSplit.train, batch_size, shuffle, num_workers, pin_memory)

    def get_val_loader(self, batch_size=10, shuffle=False, num_workers=8, pin_memory=False):
        return self._get_loader(DatasetSplit.val, batch_size, shuffle, num_workers, pin_memory)

    def get_test_loader(self, batch_size=10, shuffle=False, num_workers=8, pin_memory=False):
        return self._get_loader(DatasetSplit.test, batch_size, shuffle, num_workers, pin_memory)


if __name__ == "__main__":
    from preprocess import Preprocess
    directory = '/home/ds/Data/academic/dataset_v2'
    dataset = VideoDataset(directory)
    for i in range(0, 2700, 60):
        dataset[i]
        # print(dataset[i])
    # transform = Preprocess.get_transform()

    # loader = VideoLoader(directory, transform)

    # for idx, (data, label) in enumerate(loader.get_train_loader()):
    #     print(label)
    #     break
