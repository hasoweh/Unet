import glob
import os
import torch
import rasterio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class HedgeDataset(Dataset):
    """Hedgerow dataset class.
    """
    
    def __init__(self, root, img_folder, mask_folder, filetype = 'png', train = True):
        """
        
        Parameters
        ----------
        root : str
            Root folder containing the json file.
        img_folder : str
            Folder containing image files.
        mask_folder : str
            Folder containing mask files
        transform : callable, optional
            Optional transform to be applied on a sample.
            Should contain a list of transformations that will
            be applied to each image.
        """
        if train:
            im_path = os.path.join(root, img_folder, 'train')
            mk_path = os.path.join(root, mask_folder, 'train')
        else:
            im_path = os.path.join(root, img_folder, 'val')
            mk_path = os.path.join(root, mask_folder, 'val')
        self.root = root
        
        self.mask_folder = mk_path
        self.hedgeData = glob.glob(os.path.join(im_path, '*.%s' % filetype))
        self.maskData = glob.glob(os.path.join(mk_path, '*.%s' % filetype))
        
        assert len(self.hedgeData) == len(self.maskData), \
        'Mismatch between masks and image files'
    
    def __len__(self):
        return len(self.hedgeData)

    def __getitem__(self, idx):
        # ensure the idx is in a list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get full file path
        img_name = self.hedgeData[idx]
        image, mask = load_image_mask(img_name, self.root, self.mask_folder)
            
        #sample = {'img': torch.tensor(image), 'label': mask}

        #if self.transform:
        #    sample = self.transform(sample)

        #return sample
        return image, mask, Path(img_name).stem

def load_image_mask(img_path, root, mask_folder):
    
    # load image file
    with rasterio.open(img_path) as f:
        img = f.read()

    # find the matching mask file
    filen = Path(img_path).name
    mask_path = os.path.join(root, mask_folder, filen)

    # load mask file
    with rasterio.open(mask_path) as f:
        mask = f.read() 

    return img.astype(np.int16), np.squeeze(mask)
