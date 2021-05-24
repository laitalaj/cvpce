import torch
from torch.utils.data import DataLoader
from torchvision import ops as tvops

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
        if not len(boxes):
            return boxes, torch.empty((0, 3, datautils.CLASSIFICATION_IMAGE_SIZE, datautils.CLASSIFICATION_IMAGE_SIZE))
        return boxes, torch.stack([datautils.resize_for_classification(image[:, y1:y2, x1:x2]) for x1, y1, x2, y2 in boxes.to(dtype=torch.long)])

class Classifier:
    def __init__(self, encoder, sample_set, device=torch.device('cuda'), emb_device=torch.device('cuda'), batch_size=32, num_workers=8, k=1, load=None, verbose=False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.emb_device = emb_device
        self.k = k

        self.encoder = encoder

        if load is None:
            self.embedding, self.annotations = self.build_index(sample_set, verbose)
        else:
            self.embedding, self.annotations = self.load_index(load)
    def build_index(self, sample_set, verbose=False):
        loader = DataLoader(sample_set,
            batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=datautils.gp_annotated_collate_fn, pin_memory=True)
        embedding = torch.empty((0, self.encoder.embedding_size), dtype=torch.float, device=self.emb_device)
        annotations = []
        for i, (imgs, _, _, anns) in enumerate(loader):
            if verbose and i % 100 == 0:
                print(i)
            emb = self.encoder(imgs.to(device=self.device)).detach().to(device=self.emb_device)
            embedding = torch.cat((embedding, emb))
            annotations += anns
        return embedding, annotations
    def save_index(self, pth):
        torch.save({
            'embedding': self.embedding,
            'annotations': self.annotations,
        }, pth)
    def load_index(self, pth):
        idx = torch.load(pth)
        return idx['embedding'], idx['annotations']
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
        if image is None:
            reproj_threshold = 10
        else:
            h, w = image.shape[1:]
            reproj_threshold = min(h, w) * 0.01

        if not len(actual['boxes']):
            return 0 if len(expected['boxes']) else 1

        ge = expected['graph'] if 'graph' in expected else planograms.build_graph(expected['boxes'], expected['labels'], self.graph_threshold)
        ga = planograms.build_graph(actual['boxes'], actual['labels'], self.graph_threshold)
        matching = planograms.large_common_subgraph(ge, ga) # TODO: Possibility to use Tonioni
        if not len(matching):
            return 0
        found, missing_indices, missing_positions, missing_labels = planograms.finalize_via_ransac(
            matching, expected['boxes'], actual['boxes'], expected['labels'], actual['labels'],
            reproj_threshold=reproj_threshold,
        )
        if found is None: # --> couldn't calculate homography
            return len(matching) / len(expected['boxes'])

        if classifier is not None and image is not None and len(missing_positions):
            missing_positions = tvops.clip_boxes_to_image(missing_positions, image.shape[1:])
            valid_positions = (missing_positions[:,2] - missing_positions[:,0] > 1) & (missing_positions[:,3] - missing_positions[:,1] > 1)
            if not valid_positions.any():
                return found.sum() / len(found) # TODO: Also return which were actually missing

            missing_indices = missing_indices[valid_positions]
            missing_positions = missing_positions[valid_positions]
            missing_labels = [l for l, v in zip(missing_labels, valid_positions) if v]

            missing_imgs = torch.stack([datautils.resize_for_classification(image[:, y1:y2, x1:x2]) for x1, y1, x2, y2 in missing_positions.to(dtype=torch.long)])
            reclass_labels = classifier.classify(missing_imgs)
            for idx, expected_label, actual_label in zip(missing_indices, missing_labels, reclass_labels):
                if expected_label == actual_label[0]:
                    found[idx] = True
        return found.sum() / len(found) # TODO: Also return which were actually missing

class PlanogramEvaluator:
    def __init__(self, proposal_generator, classifier, planogram_comparator):
        self.proposal_generator = proposal_generator
        self.classifier = classifier
        self.planogram_comparator = planogram_comparator
    def evaluate(self, image, planogram):
        boxes, images = self.proposal_generator.generate_proposals_and_images(image)
        classes = [ann[0] for ann in self.classifier.classify(images)]
        compliance = self.planogram_comparator.compare(planogram,
            {'boxes': boxes.detach().cpu(), 'labels': classes},
            image, self.classifier)
        return compliance
