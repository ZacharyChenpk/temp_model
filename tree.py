import torch
import torch.nn as nn

### structure

class Tree():
	def __init__(self, root):
		self.root = root
		self.index = 0
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
		if contain_single and not (self.left and self.right):
			tmp = [self]
		if self.left:
			tmp = self.left.leaves() + tmp
		if self.right:
			tmp = tmp + self.right.leaves()
		return tmp
	def nodenum(self):
		tmp = 1
		if self.left:
			tmp = tmp + self.left.nodenum()
		if self.right:
			tmp = tmp + self.right.nodenum()
		return tmp
	def tree2graph(self):
		### just act like its name
		return
	def make_index(self, start_i = 0):
		### attach horizontial index to the root node of its subtree
		if self.left == False and self.right == False:
			self.index = start_i
			return start_i
		nodes = start_i
		if self.left:
			nodes = self.left.make_index(start_i) + 1
		self.index = nodes
		if self.right:
			return self.right.make_index(nodes + 1)
		return nodes



