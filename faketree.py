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

thetree2 = Tree('<start>')
thetree2.insert_son(0, 'have')
thetree2.insert_son(0, '.')
thetree2.make_index()
print_tree(thetree2, True)
thetree2.insert_son(0, 'I')
thetree2.insert_son(0, 'an')
thetree2.insert_son(2, 'apple')
thetree2.insert_son(2, '<end>')
thetree2.make_index()
print_tree(thetree2)
thetree2.insert_son(0, '<end>')
thetree2.insert_son(0, '<end>')
thetree2.insert_son(2, '<end>')
thetree2.insert_son(2, '<end>')
thetree2.insert_son(4, '<end>')
thetree2.insert_son(4, '<end>')
thetree2.make_index()
print_tree(thetree2, True)

thetree3 = Tree('<start>')
thetree3.insert_son(0,'you')
thetree3.insert_son(0,'banana')
thetree3.make_index()
print_tree(thetree3, True)
thetree3.insert_son(0,'Maybe')
thetree3.insert_son(0,'have')
thetree3.make_index()
thetree3.insert_son(4,'a')
thetree3.insert_son(4,'.')
thetree3.make_index()
print_tree(thetree3, True)
thetree3.insert_son(0,'<end>')
thetree3.insert_son(0,'<end>')
thetree3.insert_son(2,'<end>')
thetree3.insert_son(2,'<end>')
thetree3.insert_son(4,'<end>')
thetree3.insert_son(4,'<end>')
thetree3.insert_son(6,'<end>')
thetree3.insert_son(6,'<end>')
thetree3.make_index()
print_tree(thetree3, True)

thetree4 = Tree('<start>')
thetree4.insert_son(0, 'have')
thetree4.insert_son(0, '.')
thetree4.make_index()
print_tree(thetree4, True)
thetree4.insert_son(0, 'I')
thetree4.insert_son(0, 'an')
thetree4.insert_son(2, 'apple')
thetree4.insert_son(2, '<end>')
thetree4.make_index()
print_tree(thetree4)
thetree4.insert_son(0, '<end>')
thetree4.insert_son(0, '<end>')
thetree4.insert_son(2, '<end>')
thetree4.insert_son(2, '<end>')
thetree4.insert_son(4, '<end>')
thetree4.insert_son(4, '<end>')
thetree4.make_index()
print_tree(thetree4, True)

thetree5 = Tree('<start>')
thetree5.insert_son(0,'you')
thetree5.insert_son(0,'banana')
thetree5.make_index()
print_tree(thetree5, True)
thetree5.insert_son(0,'Maybe')
thetree5.insert_son(0,'have')
thetree5.make_index()
thetree5.insert_son(4,'a')
thetree5.insert_son(4,'.')
thetree5.make_index()
print_tree(thetree5, True)
thetree5.insert_son(0,'<end>')
thetree5.insert_son(0,'<end>')
thetree5.insert_son(2,'<end>')
thetree5.insert_son(2,'<end>')
thetree5.insert_son(4,'<end>')
thetree5.insert_son(4,'<end>')
thetree5.insert_son(6,'<end>')
thetree5.insert_son(6,'<end>')
thetree5.make_index()
print_tree(thetree5, True)

thetree6 = Tree('<start>')
thetree6.insert_son(0, 'have')
thetree6.insert_son(0, '.')
thetree6.make_index()
print_tree(thetree6, True)
thetree6.insert_son(0, 'I')
thetree6.insert_son(0, 'an')
thetree6.insert_son(2, 'apple')
thetree6.insert_son(2, '<end>')
thetree6.make_index()
print_tree(thetree6)
thetree6.insert_son(0, '<end>')
thetree6.insert_son(0, '<end>')
thetree6.insert_son(2, '<end>')
thetree6.insert_son(2, '<end>')
thetree6.insert_son(4, '<end>')
thetree6.insert_son(4, '<end>')
thetree6.make_index()
print_tree(thetree6, True)

thetree7 = Tree('<start>')
thetree7.insert_son(0,'you')
thetree7.insert_son(0,'banana')
thetree7.make_index()
print_tree(thetree7, True)
thetree7.insert_son(0,'Maybe')
thetree7.insert_son(0,'have')
thetree7.make_index()
thetree7.insert_son(4,'a')
thetree7.insert_son(4,'.')
thetree7.make_index()
print_tree(thetree7, True)
thetree7.insert_son(0,'<end>')
thetree7.insert_son(0,'<end>')
thetree7.insert_son(2,'<end>')
thetree7.insert_son(2,'<end>')
thetree7.insert_son(4,'<end>')
thetree7.insert_son(4,'<end>')
thetree7.insert_son(6,'<end>')
thetree7.insert_son(6,'<end>')
thetree7.make_index()
print_tree(thetree7, True)

thetree8 = Tree('<start>')
thetree8.insert_son(0, 'have')
thetree8.insert_son(0, '.')
thetree8.make_index()
print_tree(thetree8, True)
thetree8.insert_son(0, 'I')
thetree8.insert_son(0, 'an')
thetree8.insert_son(2, 'apple')
thetree8.insert_son(2, '<end>')
thetree8.make_index()
print_tree(thetree8)
thetree8.insert_son(0, '<end>')
thetree8.insert_son(0, '<end>')
thetree8.insert_son(2, '<end>')
thetree8.insert_son(2, '<end>')
thetree8.insert_son(4, '<end>')
thetree8.insert_son(4, '<end>')
thetree8.make_index()
print_tree(thetree8, True)

thetree9 = Tree('<start>')
thetree9.insert_son(0,'you')
thetree9.insert_son(0,'banana')
thetree9.make_index()
print_tree(thetree9, True)
thetree9.insert_son(0,'Maybe')
thetree9.insert_son(0,'have')
thetree9.make_index()
thetree9.insert_son(4,'a')
thetree9.insert_son(4,'.')
thetree9.make_index()
print_tree(thetree9, True)
thetree9.insert_son(0,'<end>')
thetree9.insert_son(0,'<end>')
thetree9.insert_son(2,'<end>')
thetree9.insert_son(2,'<end>')
thetree9.insert_son(4,'<end>')
thetree9.insert_son(4,'<end>')
thetree9.insert_son(6,'<end>')
thetree9.insert_son(6,'<end>')
thetree9.make_index()
print_tree(thetree9, True)

thetree10 = Tree('<start>')
thetree10.insert_son(0, 'have')
thetree10.insert_son(0, '.')
thetree10.make_index()
print_tree(thetree10, True)
thetree10.insert_son(0, 'I')
thetree10.insert_son(0, 'an')
thetree10.insert_son(2, 'apple')
thetree10.insert_son(2, '<end>')
thetree10.make_index()
print_tree(thetree10)
thetree10.insert_son(0, '<end>')
thetree10.insert_son(0, '<end>')
thetree10.insert_son(2, '<end>')
thetree10.insert_son(2, '<end>')
thetree10.insert_son(4, '<end>')
thetree10.insert_son(4, '<end>')
thetree10.make_index()
print_tree(thetree10, True)

torch.save([thetree, thetree2, thetree3, thetree4, thetree5, thetree6, thetree7, thetree8, thetree9, thetree10], os.path.join('fakedata', 'test_tree.txt'))