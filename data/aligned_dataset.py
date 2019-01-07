import os.path
import random
from data.base_dataset import BaseDataset, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import pickle


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    """
        1. Split the dataroot as opt.dataroot_A and opt.dataroot_B
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.A_paths = self._get_paths(datamark="A")
        self.B_paths = self._get_paths(datamark="B")

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

        # input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        # output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        # we manually crop and flip in __getitem__ to make sure we apply the same crop and flip for image A and B
        # we disable the cropping and flipping in the function get_transform

        self.transform_A = get_transform(opt, grayscale=False, crop=False, flip=True)
        self.transform_B = get_transform(opt, grayscale=False, crop=False, flip=True)

    def _get_paths(self, datamark):
        dataroot = getattr(self, "root_%s" % datamark)
        dataset_name = getattr(self, "dataset_%s" % datamark)
        with open(
                os.path.join(dataroot, "%s_train_paths.pickle" % dataset_name),
                'r') as f:
            data_paths = pickle.load(f)

        return data_paths

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = os.path.join(self.root_A, self.A_paths[index])
        B_path = os.path.join(self.root_B, random.choice(self.B_paths))

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        A = self.transform_A(A)
        B = self.transform_B(B)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
