import numpy as np

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class SpoilDataset(CustomDataset):

    CLASSES = ('spoil',)

    def load_annotations(self, ann_file):
        
        self.cat_ids = [1,]
        self.cat2label = {
            1: 1
        }
        img_infos = np.load(ann_file)
        self.img_ids = range(len(img_infos))
        #img_infos = []
        # for i in self.img_ids:
        #     img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        #img_id = self.img_infos[idx]['id']
        #ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.img_infos[idx]['label'] #self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        #ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            # if self.img_ids[i] not in ids_with_ann:
            #     continue
            # if min(img_info['width'], img_info['height']) >= min_size:
            valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            # if ann.get('ignore', False):
            #     continue
            x1, y1, w, h = int(ann[1]), int(ann[2]), int(ann[3])- int(ann[1]) + 1, int(ann[4]) - int(ann[2]) + 1
            if w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann[0] == 'ignore':
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                if ann[0] == 'spoil':
                    gt_labels.append(1)
                else:
                    gt_labels.append(int(ann[0]))
                #gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
