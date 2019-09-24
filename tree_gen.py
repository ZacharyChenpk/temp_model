import argparse
import tree
import os
import torch
from tree import Tree, print_tree

parser = argparse.ArgumentParser(description='Tree generator with given strategy')
parser.add_argument('--data', type=str, default='../training/tokenized-deende.txt',
					help='location of the data')
parser.add_argument('--cuda', action='store_false',
					help='use CUDA')
parser.add_argument('--save', type=str, default='../training/tokenized-deenen-tree',
					help='path to save the generated tree')
parser.add_argument('--strategy', type=str, default='MID',
					help='the strategy to generate the trees(L2R, R2L, MID, ...)')

args = parser.parse_args()

def sen2tree(sen, strategy, flag=False):
	if len(sen)==1:
		thetree = Tree(sen[0])
		thetree.left = Tree('<end>')
		thetree.right = Tree('<end>')
		thetree.make_index()
		return thetree
	elif len(sen)==0:
		return Tree('<end>')
	if args.strategy == 'L2R':
		if not flag:
			thetree = Tree('<start>')
			thetree.left = Tree('<end>')
			thetree.right = sen2tree(sen, strategy, True)
			thetree.make_index()
			print_tree(thetree)
			return thetree
		else:
			thetree = Tree(sen[0])
			thetree.left = Tree('<end>')
			thetree.right = sen2tree(sen[1:], strategy, True)
			return thetree
	elif args.strategy == 'R2L':
		if not flag:
			thetree = Tree('<start>')
			thetree.right = Tree('<end>')
			thetree.left = sen2tree(sen, strategy, True)
			thetree.make_index()
			print_tree(thetree)
			return thetree
		else:
			thetree = Tree(sen[-1])
			thetree.right = Tree('<end>')
			thetree.left = sen2tree(sen[0:len(sen)-1], strategy, True)
			return thetree
	elif args.strategy == 'MID':
		mid = (len(sen)+1)//2
		if flag:
			mid = len(sen)//2
		root = '<start>'
		if flag:
			root = sen[mid]
		thetree = Tree(root)
		if flag:
			thetree.right = sen2tree(sen[mid+1:], strategy, True)
		else:
			thetree.right = sen2tree(sen[mid:], strategy, True)	
		thetree.left = sen2tree(sen[0:mid], strategy, True)
		#thetree.right = sen2tree(sen[mid:], strategy, True)
		if not flag:
			thetree.make_index()
		return thetree

with open(args.data) as f:
	outputs = [sen2tree(line.split(), args.strategy) for line in f.readlines()]
	torch.save(outputs, args.save)
