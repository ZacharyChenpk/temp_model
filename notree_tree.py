import torch
import torch.nn as nn
import random
import copy
from GCN import Graph

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
        self.L_able = []
        self.R_able = []

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
        return tmp

    ### Return its leaves
    ### If contain_single, it will return the single-son node too
    def leaves(self, contain_single=False):
        if (not self.left) and (not self.right) and not(self.root == '<end>'):
            return [self]
        tmp = []
        if contain_single and (self.root != '<end>') and (self.left == False or self.right == False):
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
    def insert_son(self, father_index, son_root, Training = True):
        if self.index == father_index:
            if (not self.left) and (len(self.L_able)>0 or not Training):
                self.left = Tree(son_root)
                if son_root != '<end>' and Training:
                    i = self.L_able.index(son_root)
                    self.left.L_able = self.L_able[0:i] if i>0 else []
                    self.left.R_able = self.L_able[i+1:] if i+1<len(self.L_able) else []
            elif not self.right: 
                self.right = Tree(son_root)
                if son_root != '<end>' and Training:
                    i = self.R_able.index(son_root)
                    self.right.L_able = self.R_able[0:i] if i>0 else []
                    self.right.R_able = self.R_able[i+1:] if i+1<len(self.R_able) else []
            else:
                return False
            return True
        else:
            if self.left == False or self.left.insert_son(father_index, son_root) == False:
                return self.right and self.right.insert_son(father_index, son_root)
            return True

    def find_able_pos(self, word, flag=False):
        if not flag:
            self.make_index(0)
        if self.left and self.right:
            return self.left.find_able_pos(word,flag=True) + self.right.find_able_pos(word,flag=True)
        tmp = []
        if self.left == False and (word in self.L_able):
            tmp = [self.index]
        if self.right == False and (word in self.R_able):
            tmp = [self.index]
        if self.left:
            tmp = self.left.find_able_pos(word,flag=True) + tmp
        if self.right:
            tmp = tmp + self.right.find_able_pos(word,flag=True)
        return tmp

    def able_words(self):
        tmp = []
        if self.left == False:
            tmp = self.L_able
        if self.right == False:
            tmp += self.R_able
        if self.left:
            tmp = self.left.able_words() + tmp
        if self.right:
            tmp = tmp + self.right.able_words()
        return list(set(tmp))


def behave_seq_gen(sen):
    cur_tree = Tree('<start>')
    l = len(sen)
    ran_split = random.randint(1,l)
    cur_tree.L_able = sen[0:ran_split] if ran_split>0 else []
    cur_tree.R_able = sen[ran_split:] if ran_split<l else []
    ans_ind = []
    choose_words = []
    trees_before_insert = []
    if l == 1:
        ans_ind = [0]
        trees_before_insert = [cur_tree]
        choose_words = sen
        return ans_ind, choose_words, trees_before_insert
    while True:
        alleaves = cur_tree.leaves(True)
        judge = lambda x: True if ((x.left == False and len(x.L_able)>0) or (x.right == False and len(x.R_able)>0)) else False
        leaves = [leave for leave in alleaves if judge(leave)]
        if len(leaves) == 0:
            return ans_ind, choose_words, trees_before_insert
        ran_split = random.randint(0,len(leaves)-1) if len(leaves)>1 else 0
        ran_parent = leaves[ran_split]
        ans_ind.append(ran_parent.index)
        trees_before_insert.append(copy.deepcopy(cur_tree))
        ables = ran_parent.L_able if (ran_parent.left == False and len(ran_parent.L_able)>0) else ran_parent.R_able
        # print(ran_parent.root,':',ables)
        ran_split = random.randint(0,len(ables)-1) if len(ables)>1 else 0
        choose_words.append(ables[ran_split])
        cur_tree.insert_son(ran_parent.index, ables[ran_split])
        cur_tree.make_index()


'''
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
'''
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