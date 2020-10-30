import argparse
from nltk.tree import Tree


def dependencyGraph(parseTree):
    if parseTree.label() == 'ROOT':
        # root, index = dependencyGraph(parseTree[0])
        root = dependencyGraph(parseTree[0])
        # print('root(ROOT-0, ' + root + '-' + index + ')')
        print('root(ROOT, ' + root+')')
        return root
    elif parseTree.label() == 'S':
        # root_NP, index_NP = dependencyGraph(parseTree[0])
        root_NP = dependencyGraph(parseTree[0])
        # root_VP, index_VP = dependencyGraph(parseTree[1])
        root_VP = dependencyGraph(parseTree[1])
        # print('nsubj(' + root_VP + '-' + index_VP + ', ' + root_NP + '-' + index_NP + ')')
        print('nsubj(' + root_VP + ', ' + root_NP + ')')
        return root_VP
    elif parseTree.label() == 'NP':
        # root_JJ, index_JJ = dependencyGraph(parseTree[0])
        root_JJ = dependencyGraph(parseTree[0])
        if len(parseTree) == 1:
            return root_JJ
        # root_NN, index_NN = dependencyGraph(parseTree[1])
        root_NN = dependencyGraph(parseTree[1])
        # print('amod(' + root_NN + '-' + index_NN + ', ' + root_JJ + '-' + index_JJ + ')')
        print('amod(' + root_NN + ', ' + root_JJ + ')')
        return root_NN
    elif parseTree.label() == 'JJ':
        return parseTree[0]
    elif parseTree.label() == 'NNS':
        return parseTree[0]
    elif parseTree.label() == 'VP':
        # root_VBD, index_VBD = dependencyGraph(parseTree[0])
        root_VBD = dependencyGraph(parseTree[0])
        # root_NP, index_NP = dependencyGraph(parseTree[1])
        root_NP = dependencyGraph(parseTree[1])
        print('obj(' + root_VBD + ', ' + root_NP + ')')
        return root_VBD
    elif parseTree.label() == 'VBD':
        return parseTree[0]
    elif parseTree.label() == 'NN':
        return parseTree[0]
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input to the CP2DP transformer")
    parser.add_argument('--CP', dest='CP', help="provide path to the CP txt file")

    args = parser.parse_args()
    CP = args.CP
    CPfile = open(CP, 'r')
    parseTree = Tree.fromstring(CPfile.read())
    G = dependencyGraph(parseTree)
