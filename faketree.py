import tree
import os
import torch
from tree import Tree, print_tree

thetree = Tree('<start>')
thetree.insert_son(0,'you')
thetree.insert_son(0,'banana')
thetree.make_index()
print_tree(thetree, True)
thetree.insert_son(0,'Maybe')
thetree.insert_son(0,'have')
thetree.make_index()
thetree.insert_son(4,'a')
thetree.insert_son(4,'.')
thetree.make_index()
print_tree(thetree, True)
thetree.insert_son(0,'<end>')
thetree.insert_son(0,'<end>')
thetree.insert_son(2,'<end>')
thetree.insert_son(2,'<end>')
thetree.insert_son(4,'<end>')
thetree.insert_son(4,'<end>')
thetree.insert_son(6,'<end>')
thetree.insert_son(6,'<end>')
thetree.make_index()
print_tree(thetree, True)

torch.save([thetree], os.path.join('fakedata', 'test_tree.txt'))