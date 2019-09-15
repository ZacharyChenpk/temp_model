import torch
import torch.nn as nn
import random
import copy
from fakegcn import Graph

### structure

class Tree():
	### init 
	### The MASK is used to copy trees while training the position and word choser
	### We used the processing tree as the input of training
	### Its word is contained in ROOT
	def __init__(self, root):
		self.root = root
		self.index = 0
		self.left = False
		self.right = False
		### 0: don't copy its son
		### 1: only copy left son if it have
		### 2: copy both left and right sons
		self.mask = 0

	def __deepcopy__(self, memo):
		if memo is None:
			memo = {}
		dup = Tree(self.root)
		dup.index = self.index
		if self.left and self.mask > 0:
			dup.left = copy.deepcopy(self.left)
		if self.right and self.mask > 1:
			dup.right = copy.deepcopy(self.right)
		return dup

	### Return the result (sentence?) of horizontal scanning
	def horizontal_scan(self, contain_end = True):
		flag = bool(contain_end or (not self.root == '<end>'))
		tmp = [self.root] if flag else []
		if self.left:
			tmp = self.left.horizontal_scan(contain_end)
			if flag:
				tmp.append(self.root)
		if self.right:
			tmp.extend(self.right.horizontal_scan(contain_end))

	### Return its leaves
	### If contain_single, it will return the single-son node too
	def leaves(self, contain_single=False):
		if (not self.left) and (not self.right) and not(self.root == '<end>'):
			return [self]
		tmp = []
		if contain_single and (self.left == False or self.right == False):
			tmp = [self]
		if self.left:
			tmp = self.left.leaves(contain_single) + tmp
		if self.right:
			tmp = tmp + self.right.leaves(contain_single)
		return tmp

	def nodenum(self):
		tmp = 1
		if self.left:
			tmp = tmp + self.left.nodenum()
		if self.right:
			tmp = tmp + self.right.nodenum()
		return tmp

	def tree2graph(self, sen_encoder, dictionary, nodedim):
		### just act like its name, waiting to finish
		nodenum = self.make_index() + 1
		the_graph = Graph(nodenum, nodedim, nodedim)
		the_graph.match_tree(self, sen_encoder, dictionary)
		return the_graph

	### Attach horizontial index to the root node of its subtree
	### Return the max index in the subtree
	def make_index(self, start_i = 0):
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

	### Find the index of specific node with given word (theroot)
	def find_index(self, theroot):
		a = self.left.find_index(theroot) if self.left else False
		if a is not False:
			return a
		if self.root == theroot:
			return self.index
		a = self.right.find_index(theroot) if self.right else False
		if a is not False:
			return a
		return False

	### Insert a node with given word
	def insert_son(self, father_index, son_root):
		if self.index == father_index:
			if not self.left:
				self.left = Tree(son_root)
			elif not self.right: 
				self.right = Tree(son_root)
			else:
				return False
			return True
		else:
			if self.left == False or self.left.insert_son(father_index, son_root) == False:
				return self.right and self.right.insert_son(father_index, son_root)
			return True

def refresh_mask(tree):
	tree.mask = 0
	if tree.left:
		refresh_mask(tree.left)
	if tree.right:
		refresh_mask(tree.right)

def random_seq(tree):
	dest = tree.nodenum()
	candidate = [tree]
	seq = []
	indseq = []
	wordseq = []
	ansseq = []
	treeseq = [copy.deepcopy(tree)]
	tmp = copy.deepcopy(tree)
	tmp.make_index()
	while len(candidate) > 0:
		flag = False
		a = random.choice(candidate)
		seq.append(a)
		wordseq.append(a.root)
		indseq.append(tmp.find_index(a.root))
		if a.mask == 1 and a.right:
			ansseq.append(a.right.root)
		elif a.left:
			ansseq.append(a.left.root)
		if (a.left is not False) and not ((a.left in seq) or (a.left in candidate) or a.mask == 1):
			if a.left.root != '<end>':
				candidate.append(a.left)
			a.mask = 1
			if a.right is not False:
				flag = True
		elif (a.right is not False):
			if a.right.root != '<end>':
				candidate.append(a.right)
			a.mask = 2
		if flag == False:
			candidate.remove(a)
		tmp = copy.deepcopy(tree)
		tmp.make_index()
		treeseq.append(tmp)
		#print_tree(tmp, True)
		
	return seq, indseq, wordseq, treeseq[:-1], ansseq

def _build_tree_string(root, show_index=False):
	# SOURCE: https://github.com/joowani/binarytree
	if root is None:
		return [], 0, 0, 0
	if root is False:
		return [], 0, 0, 0

	line1 = []
	line2 = []
	if show_index:
		node_repr = '{}-{}'.format(root.index, root.root)
	else:
		node_repr = str(root.root)

	new_root_width = gap_size = len(node_repr)

	# Get the left and right sub-boxes, their widths, and root repr positions
	l_box, l_box_width, l_root_start, l_root_end = \
		_build_tree_string(root.left, show_index)
	r_box, r_box_width, r_root_start, r_root_end = \
		_build_tree_string(root.right, show_index)

	# Draw the branch connecting the current root node to the left sub-box
	# Pad the line with whitespaces where necessary
	if l_box_width > 0:
		l_root = (l_root_start + l_root_end) // 2 + 1
		line1.append(' ' * (l_root + 1))
		line1.append('_' * (l_box_width - l_root))
		line2.append(' ' * l_root + '/')
		line2.append(' ' * (l_box_width - l_root))
		new_root_start = l_box_width + 1
		gap_size += 1
	else:
		new_root_start = 0

	# Draw the representation of the current root node
	line1.append(node_repr)
	line2.append(' ' * new_root_width)

	# Draw the branch connecting the current root node to the right sub-box
	# Pad the line with whitespaces where necessary
	if r_box_width > 0:
		r_root = (r_root_start + r_root_end) // 2
		line1.append('_' * r_root)
		line1.append(' ' * (r_box_width - r_root + 1))
		line2.append(' ' * r_root + '\\')
		line2.append(' ' * (r_box_width - r_root))
		gap_size += 1
	new_root_end = new_root_start + new_root_width - 1

	# Combine the left and right sub-boxes with the branches drawn above
	gap = ' ' * gap_size
	new_box = [''.join(line1), ''.join(line2)]
	for i in range(max(len(l_box), len(r_box))):
		l_line = l_box[i] if i < len(l_box) else ' ' * l_box_width
		r_line = r_box[i] if i < len(r_box) else ' ' * r_box_width
		new_box.append(l_line + gap + r_line)

	# Return the new box, its width and its root repr positions
	return new_box, len(new_box[0]), new_root_start, new_root_end

def print_tree(tree, show_index=False):
	lines = _build_tree_string(tree, show_index)[0]
	print('tree word sequence:')
	for i in tree.leaves(contain_single=False):
		print(i.root, end=' ')
	print('\n' + '\n'.join((line.rstrip() for line in lines)))