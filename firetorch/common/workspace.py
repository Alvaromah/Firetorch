import os
import cv2
import datetime

import torch

class Workspace():
    def __init__(self, root, category=None, pattern='%m%d_%H%M%S'):
        self.path = root
        if category:
            self.path = os.path.join(root, category)
        if pattern:
            self.key = datetime.datetime.now().strftime(pattern)
            self.path = os.path.join(self.path, self.key)

    def write_file(self, txt, *route):
        fn = os.path.join(self.path, *route)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, 'a', encoding='UTF-8') as fp:
            fp.write(txt)

    def save_image(self, src, *route, resize=None):
        '''
            Save image using cv2.
            Supported config: (H, W), (H, W, 1), (H, W, 3)
            Types: uint8, float32 0-255
            Examples:
                save_image(src, 'filename.png')
                save_image(src, 'path', 'filename.png')
        '''
        fn = os.path.join(self.path, *route)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        if isinstance(type(src), type(torch.Tensor)):
            src = src.detach().cpu().numpy()
        if resize:
            src = cv2.resize(src, (resize, resize))
        cv2.imwrite(fn, src)

    def ensure_path(self):
        os.makedirs(self.path, exist_ok=True)

    def create_dir(self, *route):
        path = self.get_path(*route)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    def get_path(self, *route):
        return os.path.join(self.path, *route)
