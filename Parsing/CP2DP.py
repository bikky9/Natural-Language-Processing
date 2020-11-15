import argparse
from nltk.tree import Tree
"""
Grammar used:(* denotes head)
ROOT -> *S
S -> NP *VP | *VP
NP -> JJ *NNS | *NNS | NN *NN | *NP SBAR | PRP$ *NNS | *NP PP | *NP VP
SBAR -> WHNP *S
WHNP -> *WP
VP -> *VBD NP | VBD *VP | *VBN NP | *VBD ADVP NP | *VBG
ADVP -> *RB
PP -> IN *NP
"""


def dependencyGraph(parseTree):
    if parseTree.label() == 'ROOT':
        root_ROOT, index_ROOT, type_ROOT = dependencyGraph(parseTree[0])
        print('root(ROOT-0, ' + root_ROOT + '-' + index_ROOT + ')')
        return root_ROOT, index_ROOT, parseTree.label()
    elif parseTree.label() == 'S':
        root_l, index_l, type_l = dependencyGraph(parseTree[0])
        if len(parseTree) == 1:
            return root_l, index_l, parseTree.label()
        root_r, index_r, type_r = dependencyGraph(parseTree[1])
        if type_l == "NP" and type_r == "VP":
            print('nsubj(' + root_r + '-' + index_r + ', ' + root_l + '-' + index_l + ')')
            return root_r, index_r, parseTree.label()
    elif parseTree.label() == 'NP':
        root_l, index_l, type_l = dependencyGraph(parseTree[0])
        if len(parseTree) == 1:
            return root_l, index_l, parseTree.label()

        root_r, index_r, type_r = dependencyGraph(parseTree[1])
        if type_l == "JJ" and (type_r == "NN" or type_r == "NNS"):
            print('amod(' + root_r + '-' + index_r + ', ' + root_l + '-' + index_l + ')')
            return root_r, index_r, parseTree.label()
        elif type_l == "NN" and (type_r == "NN" or type_r == "NNS"):
            print('compound(' + root_r + '-' + index_r + ', ' + root_l + '-' + index_l + ')')
            return root_r, index_r, parseTree.label()
        elif type_l == "NP" and type_r == "SBAR":
            print('acl:relcl(' + root_l + '-' + index_l + ', ' + root_r + '-' + index_r + ')')
            return root_l, index_l, parseTree.label()
        elif type_l == "PRP$" and (type_r == "NN" or type_r == "NNS"):
            print('nmod:poss(' + root_r + '-' + index_r + ', ' + root_l + '-' + index_l + ')')
            return root_r, index_r, parseTree.label()
        elif type_l == "NP" and type_r == "VP":
            print('acl(' + root_l + '-' + index_l + ', ' + root_r + '-' + index_r + ')')
            return root_l, index_l, parseTree.label()
        elif type_l == "NP" and type_r == "PP":
            print('nmod(' + root_l + '-' + index_l + ', ' + root_r + '-' + index_r + ')')
            return root_l, index_l, parseTree.label()
    elif parseTree.label() == 'PP':
        root_l, index_l, type_l = dependencyGraph(parseTree[0])
        root_r, index_r, type_r = dependencyGraph(parseTree[1])
        if type_l == "IN" and type_r == "NP":
            print('case(' + root_r + '-' + index_r + ', ' + root_l + '-' + index_l + ')')
            return root_r, index_r, parseTree.label()
    elif parseTree.label() == 'SBAR':
        root_l, index_l, type_l = dependencyGraph(parseTree[0])
        root_r, index_r, type_r = dependencyGraph(parseTree[1])
        if type_l == "WHNP" and type_r == 'S':
            print('nsubj(' + root_r + '-' + index_r + ', ' + root_l + '-' + index_l + ')')
            return root_r, index_r, parseTree.label()
    elif parseTree.label() == "WHNP":
        root_l, index_l, type_l = dependencyGraph(parseTree[0])
        return root_l, index_l, parseTree.label()
    elif parseTree.label() == 'VP':
        root_l, index_l, type_l = dependencyGraph(parseTree[0])
        if len(parseTree) == 1:
            if type_l == "VBD" or type_l == 'VBG':
                return root_l, index_l, parseTree.label()
        root_r, index_r, type_r = dependencyGraph(parseTree[1])
        if len(parseTree) == 2:
            if (type_l == 'VBD' or type_l == 'VBN') and type_r == 'NP':
                print('obj(' + root_l + '-' + index_l + ', ' + root_r + '-' + index_r + ')')
                return root_l, index_l, parseTree.label()
            if type_l == 'VBD' and type_r == 'VP':
                print('aux(' + root_r + '-' + index_r + ', ' + root_l + '-' + index_l + ')')
                return root_r, index_r, parseTree.label()
        elif len(parseTree) == 3:
            root_e, index_e, type_e = dependencyGraph(parseTree[2])
            if type_l == 'VBD' and type_r == 'ADVP' and type_e == 'NP':
                print('advmod(' + root_l + '-' + index_l + ', ' + root_r + '-' + index_r + ')')
                print('obj(' + root_l + '-' + index_l + ', ' + root_e + '-' + index_e + ')')
                return root_l, index_l, parseTree.label()
    elif parseTree.label() == 'ADVP':
        root_l, index_l, type_l = dependencyGraph(parseTree[0])
        return root_l, index_l, parseTree.label()
    elif parseTree.label() in ['JJ', 'NNS', 'NN', 'VBD', 'VBN', 'VBG', 'WP', 'IN', 'PRP$', 'RB']:
        global index
        index += 1
        return parseTree[0], str(index), parseTree.label()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input to the CP2DP transformer")
    parser.add_argument('--CP', dest='CP', help="provide path to the CP txt file")

    args = parser.parse_args()
    CP = args.CP
    CPfile = open(CP, 'r')
    parseTree = Tree.fromstring(CPfile.read())
    index = 0
    G = dependencyGraph(parseTree)
