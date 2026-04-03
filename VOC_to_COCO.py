import os
import json
from xml.etree import ElementTree as ET
from collections import defaultdict
from tqdm import tqdm


class VocToCoco:

    def __init__(self, voc_gt_dir: str, img_dir: str, output_coco_path: str) -> None:
        self.voc_gt_dir = voc_gt_dir
        self.img_dir = img_dir  # 图片目录
        self.output_coco_path = output_coco_path
        self.categories_count = 1
        self.images = []
        self.categories = {}
        self.annotations = []
        self.data = defaultdict(list)

        # 定义你希望的类别顺序
        self.desired_order = ["drink","smoke","phone by hand","hand on the wheel","play mobile by hand","face","eye","mouth"]

    # 图片处理
    def images_handle(self, root: ET.Element, img_id: int, img_file: str) -> None:
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)

        self.images.append({
            'id': int(img_id),
            'file_name': img_file,  # 使用实际图片文件名
            'height': height,
            'width': width,
        })

    # 标签转换 - 按照指定顺序处理
    def categories_handle(self, category: str) -> None:
        if category not in self.categories:
            # 按照desired_order中的顺序分配ID
            if category in self.desired_order:
                category_id = self.desired_order.index(category) + 1
                self.categories[category] = {'id': category_id, 'name': category}

    # 标注转换
    def annotations_handle(self, bbox: ET.Element, img_id: int, category: str) -> None:
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)

        w = x2 - x1
        h = y2 - y1

        self.annotations.append({
            'id': self.categories_count,
            'image_id': int(img_id),
            'category_id': self.categories[category].get('id'),
            'bbox': [x1, y1, w, h],
            'area': w * h,
            'iscrowd': 0
        })
        self.categories_count += 1

    def parse_voc_annotation(self) -> None:
        # 获取图片目录中的所有图片文件
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
        img_files = [f for f in os.listdir(self.img_dir)
                     if any(f.lower().endswith(ext) for ext in img_extensions)]
        img_files.sort()  # 按名称排序

        for img_id, img_file in enumerate(tqdm(img_files, desc="Processing Images"), 1):
            # 找到对应的 XML 文件（假设 XML 文件名与图片名相同，只是扩展名不同）
            xml_base_name = os.path.splitext(img_file)[0]
            xml_file = os.path.join(self.voc_gt_dir, xml_base_name + '.xml')

            if not os.path.exists(xml_file):
                print(f"Warning: No XML file found for {img_file}, skipping...")
                continue

            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                self.images_handle(root, img_id, img_file)

                for obj in root.iter('object'):
                    category = obj.find('name').text
                    self.categories_handle(category)

                    bbox = obj.find('bndbox')
                    self.annotations_handle(bbox, img_id, category)

            except ET.ParseError as e:
                print(f"Error parsing XML file {xml_file}: {e}")
                continue

        # 确保categories按照desired_order排序
        sorted_categories = []
        for category_name in self.desired_order:
            if category_name in self.categories:
                sorted_categories.append(self.categories[category_name])

        # 添加可能不在desired_order中的其他类别
        for category_name, category_info in self.categories.items():
            if category_name not in self.desired_order:
                sorted_categories.append(category_info)

        self.data['images'] = self.images
        self.data['categories'] = sorted_categories
        self.data['annotations'] = self.annotations

        with open(self.output_coco_path, 'w') as f:
            json.dump(self.data, f, indent=4)

        print(f"Conversion completed. Total images: {len(self.images)}, Total annotations: {len(self.annotations)}")


if __name__ == "__main__":
    # Example usage
    voc_gt_dir = r"/home/littlecoke/Desktop/MLDDBI_13800_distraction/test/labels_xml"
    img_dir = r"/home/littlecoke/Desktop/MLDDBI_13800_distraction/test/images"
    output_coco_path = r"/home/littlecoke/Desktop/MLDDBI_13800_distraction/test/distraction_val.json"

    voc2coco = VocToCoco(voc_gt_dir, img_dir, output_coco_path)
    voc2coco.parse_voc_annotation()