import torch
import torch.nn as nn

### structure

class Tree():
	def __init__(self, root):
		self.root = root
		self.left = False
		self.right = False
	def horizontal_scan(self):
		tmp = [root]
		if self.left:
			tmp = self.left.horizontal_scan()
			tmp.append(root)
		if self.right:
			tmp.extend(self.right.horizontal_scan())
	def leaves(self, contain_single=False):
		if (not self.left) and (not self.right):
			return [self]
		tmp = []
		if contain_single and not (self.left and self.right)
			tmp = [self]
		if self.left:
			tmp = self.left.leaves() + tmp
		if self.right:
			tmp = tmp + self.right.leaves()
	