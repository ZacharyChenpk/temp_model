import torch
import torch.nn as nn
import random
import copy

### structure

class Tree():
	def __init__(self, root):
		self.root = root
		self.word = ''
		self.index = 0
		self.left = False
		self.right = False
		self.mask = 0
	def __deepcopy__(self, memo):
		if memo is None:
			memo = {}
		dup = Tree(self.root)
		dup.word = self.word
		dup.index = self.index
		if self.left and self.mask > 0:
			dup.left = copy.deepcopy(self.left)
		if self.right and self.mask > 1:
			dup.right = copy.deepcopy(self.right)
		return dup
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
		### return the max index in the subtree
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
	def find_index(self, theroot):
		a = self.left.find_index(theroot) if self.left else False
		if a:
			return a
		if self.root == theroot:
			return self.index
		a = self.right.find_index(theroot) if self.right else False
		if a:
			return a
		return False


def random_seq(t):
	dest = t.nodenum()
	candidate = [t]
	seq = []
	indseq = []
	wordseq = []
	treeseq = [copy.deepcopy(cur_tree)]
	while len(candidate) > 0:
		flag = False
		a = random.choice(candidate)
		seq.append(a)
		wordseq.append(a.word)
		if a.left and not (a.left in seq):
			candidate.append(a.left)
			a.mask = 1
			if a.right:
				flag = True
		elif a.right:
			candidate.append(a.right)
			a.mask = 2
		if flag == False:
			candidate.remove(a)
		tmp = copy.deepcopy(t)
		tmp.make_index()
		treeseq.append(tmp)
		indseq.append(tmp.find_index(a.root))
	return seq, indseq, wordseq, treeseq[:-1]
