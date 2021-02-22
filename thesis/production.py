import torch
from torch.utils.data import DataLoader

from . import datautils, utils
from .models.classification import nearest_neighbors

class Classifier:
    def __init__(self, encoder, sample_set, device=torch.device('cuda'), batch_size=32, num_workers=8, k=1):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.k = k

        self.encoder = encoder

        self.embedding, self.annotations = self.build_index(sample_set)
    def build_index(self, sample_set):
        loader = DataLoader(sample_set,
            batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=datautils.gp_annotated_collate_fn, pin_memory=True)
        embedding = torch.empty((0, 1024), dtype=torch.float, device=self.device)
        annotations = []
        for imgs, _, _, anns in loader:
            emb = self.encoder(imgs.to(device=self.device))
            embedding = torch.cat((embedding, emb))
            annotations += anns
        return embedding, annotations
    def classify(self, images):
        res = []
        for i in range(0, len(images), self.batch_size):
            batch = utils.scale_to_tanh(images[i : i+self.batch_size].to(device=self.device))
            emb = self.encoder(batch)
            nearest = nearest_neighbors(self.embedding, emb, self.k)
            if self.k == 1:
                res += [self.annotations[j] for j in nearest]
            else:
                res += [[self.annotations[j] for j in n] for n in nearest]
        return res
