import torch
from torch.utils.data import DataLoader

from . import datautils, planograms, utils
from .models.classification import nearest_neighbors

class ProposalGenerator:
    def __init__(self, detector, device = torch.device('cuda'), confidence_threshold = 0.5):
        self.detector = detector
        self.device = device
        self.condfidence_threshold = confidence_threshold
    def generate_proposals(self, image):
        res = self.detector(image[None].to(self.device))[0]
        return res['boxes'][res['scores'] > self.condfidence_threshold]
    def generate_proposals_and_images(self, image):
        boxes = self.generate_proposals(image)
        return boxes, torch.stack([datautils.resize_for_classification(image[:, y1:y2, x1:x2]) for x1, y1, x2, y2 in boxes])

class Classifier:
    def __init__(self, encoder, sample_set, device=torch.device('cuda'), emb_device=torch.device('cuda'), batch_size=32, num_workers=8, k=1):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.emb_device = emb_device
        self.k = k

        self.encoder = encoder

        self.embedding, self.annotations = self.build_index(sample_set)
    def build_index(self, sample_set):
        loader = DataLoader(sample_set,
            batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=datautils.gp_annotated_collate_fn, pin_memory=True)
        embedding = torch.empty((0, self.encoder.embedding_size), dtype=torch.float, device=self.emb_device)
        annotations = []
        for imgs, _, _, anns in loader:
            emb = self.encoder(imgs.to(device=self.device)).detach().to(device=self.emb_device)
            embedding = torch.cat((embedding, emb))
            annotations += anns
        return embedding, annotations
    def classify(self, images):
        res = []
        for i in range(0, len(images), self.batch_size):
            batch = utils.scale_to_tanh(images[i : i+self.batch_size].to(device=self.device))
            emb = self.encoder(batch).detach().to(device = self.emb_device)
            nearest = nearest_neighbors(self.embedding, emb, self.k)
            res += [[self.annotations[j] for j in n] for n in nearest]
        return res

class PlanogramComparator:
    def __init__(self, graph_threshold = 0.5):
        self.graph_threshold = graph_threshold
    def compare(self, expected, actual, image=None, classifier=None):
        ge = expected['graph'] if 'graph' in expected else planograms.build_graph(expected['boxes'], expected['labels'], self.graph_threshold)
        ga = planograms.build_graph(actual['boxes'], actual['labels'], self.graph_threshold)
        matching = planograms.large_common_subgraph(ge, ga) # TODO: Possibility to use Tonioni
        found, missing_indices, missing_positions, missing_labels = planograms.finalize_via_ransac(
            matching, expected['boxes'], actual['boxes'], expected['labels'], actual['labels']
        )

        if classifier is not None and image is not None:
            missing_imgs = torch.stack([datautils.resize_for_classification(image[:, y1:y2, x1:x2]) for x1, y1, x2, y2 in missing_positions])
            reclass_labels = classifier.classify(missing_imgs)
            for idx, expected_label, actual_label in zip(missing_indices, missing_labels, reclass_labels):
                if expected_label == actual_label:
                    found[idx] = True
        return found.sum() / len(found) # TODO: Also return which were actually missing

class PlanogramEvaluator:
    def __init__(self, proposal_generator, classifier, planogram_comparator):
        self.proposal_generator = proposal_generator
        self.classifier = classifier
        self.planogram_comparator = planogram_comparator
    def evaluate(self, image, planogram):
        boxes, images = self.proposal_generator.generate_proposals_and_images(image)
        classes = self.classifier.classify(images)
        compliance = self.planogram_comparator.compare(planogram, {'boxes': boxes, 'labels': classes}, image, self.classifier)
        return compliance
