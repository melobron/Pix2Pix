from torch.utils.data import Dataset
import cv2
from utils import *


class FacadesDataset(Dataset):
    def __init__(self, domain1='photo', domain2='segmentation', train=True, transform=None):
        super(FacadesDataset, self).__init__()

        img_dir = '../all_datasets/facades/'
        if train:
            img_dir = os.path.join(img_dir, 'train')
        else:
            img_dir = os.path.join(img_dir, 'test')

        # domain1, domain2 is one of (photo, segmentation)
        domain1_dir = os.path.join(img_dir, domain1)
        domain2_dir = os.path.join(img_dir, domain2)

        self.domain1_paths = sorted(make_dataset(domain1_dir))
        self.domain2_paths = sorted(make_dataset(domain2_dir))

        self.transform = transform

    def __getitem__(self, item):
        domain1_path = self.domain1_paths[item]
        domain2_path = self.domain2_paths[item]

        domain1_numpy = cv2.cvtColor(cv2.imread(domain1_path), cv2.COLOR_BGR2RGB)
        domain2_numpy = cv2.cvtColor(cv2.imread(domain2_path), cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=domain1_numpy, target=domain2_numpy)

        return {'domain1': transformed['image'], 'domain2': transformed['target']}

    def __len__(self):
        return len(self.domain1_paths)


if __name__ == "__main__":
    dataset = FacadesDataset()
    index = 399
    print(dataset.domain1_paths[index])
    print(dataset.domain2_paths[index])
