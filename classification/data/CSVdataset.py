import pandas as pd
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive
import os

def default_loader(path: str) -> Image.Image:
	"""Default image loader function"""
	try:
		with open(path, 'rb') as f:
			img = Image.open(f)
			# 立即加载图像数据到内存，避免多进程环境中的file seek错误
			img.load()  # 这一行很重要！确保图像数据完全加载到内存
			
			# 确保图像为RGB格式，无论原始格式是什么
			if img.mode != 'RGB':
				if img.mode in ('RGBA', 'LA'):
					# 处理带透明通道的图像，使用白色背景
					background = Image.new('RGB', img.size, (255, 255, 255))
					if img.mode == 'RGBA':
						background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
					else:  # LA模式
						background.paste(img, mask=img.split()[-1])
					img = background
				else:
					# 其他模式直接转换为RGB
					img = img.convert('RGB')
			return img
	except Exception as e:
		raise RuntimeError(f"Error loading image {path}: {e}")

class LaSBiRD(VisionDataset):
	"""
	LaSBiRD is a large-scale dataset for bird species classification.

	Args:
		csv_file (string): Path to the csv file with annotations.
		root (string, optional): Root directory of the dataset (can be None if filepaths are absolute).
		train (bool, optional): If True, creates dataset from training set, otherwise
			creates from test set (if applicable, current implementation does not use this).
		transform (callable, optional): A function/transform that takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
	"""

	def __init__(
			self,
			csv_file: str,
			root: Optional[str] = None,
			train: bool = True,  # train argument is kept for consistency but not used for split
			transform: Optional[Callable] = None,
			target_transform: Optional[Callable] = None,
	):
		super().__init__(root, transform=transform, target_transform=target_transform)
		self.csv_file = csv_file
		self.loader = default_loader
		self.train = train # Retained for API consistency, actual split logic might depend on CSV

		self._load_metadata()

	def _load_metadata(self):
		print(f"Loading metadata from {self.csv_file}")
		try:
			self.data = pd.read_csv(self.csv_file)
		except FileNotFoundError:
			raise RuntimeError(f"CSV file not found at {self.csv_file}")

		# Generate class names and class_to_idx mapping
		self.classes = sorted(self.data['caption'].unique())
		print(f"Found {len(self.classes)} classes")
		self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
		self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}

	def force_edit_maps(self, new_classes, new_class_to_idx, new_idx_to_class):
		self.classes = new_classes
		self.class_to_idx = new_class_to_idx
		self.idx_to_class = new_idx_to_class
		print(f"Forcely edit the class_to_idx and idx_to_class maps.\n And the total number of class is changed to {len(self.classes)}")

	def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
		sample = self.data.iloc[idx]
		img_id = sample['id']
		img_path = sample['filepath'] # Assuming filepath is absolute
		caption = sample['caption']
		target = self.class_to_idx[caption]

		try:
			img = self.loader(img_path)
		except Exception as e:
			raise RuntimeError(f"Error loading image {img_path} (id: {img_id}): {e}")

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		# 返回三个值：img_id, img, target（用户需要img_id用于测试）
		return img_id, img, target

	def __len__(self) -> int:
		return len(self.data)