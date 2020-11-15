import argparse
from collections import defaultdict
from nltk.tree import Tree


def children(head, phrase):
    if head in graph:
        childs = []
        for key in graph[head]:
            if key == "nsubj":
                if phrase == "S":
                    childs.append(Tree("NP", children(graph[head][key], "NP")))
                if phrase == "WHNP":
                    childs += children(graph[head][key], "WP")
            if key == "obj":
                if phrase == "S":
                    childs.append(
                        Tree("VP", [Tree("V", [head]), Tree("NP", children(graph[head][key], "NP"))]))
                if phrase == "VP":
                    childs.append(Tree("VP", [Tree("V", [head])]+children(graph[head][key], "VP")))
            if key == "amod":
                if len(graph[head]) == 1:
                    childs.append(Tree("JJ", [graph[head][key]]))
                    childs.append(Tree("N", [head]))
                else:
                    childs.append(Tree("NP", [Tree("JJ", [graph[head][key]]), Tree("N", [head])]))
            if key == "acl:relcl":
                childs.append(Tree("SBAR", [Tree("WHNP", children(graph[head][key], "WHNP")), Tree("S", [Tree("VP", children(graph[head][key], "VP"))])]))
            if key == "aux":
                if phrase == "VP":
                    childs += children(graph[head][key], "V")
            if key == "nmod:poss":
                childs.append(Tree("NP", [Tree("PRP$", children(graph[head][key], "PRP$")), Tree("N", [head])]))
            if key == "compound":
                if len(graph[head]) == 1:
                    childs.append(Tree("JJ", [graph[head][key]]))
                    childs.append(Tree("N", [head]))
                else:
                    childs.append(Tree("NP", [Tree("JJ", [graph[head][key]]), Tree("N", [head])]))
        return childs
    else:
        return [Tree("POS", [head])]


def constituencyTree(head):
    if head == "ROOT-0":
        return Tree("ROOT", [Tree("S", children(graph[head]["root"], "S"))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input to the CP2DP transformer")
    parser.add_argument('--DP', dest='DP', help="provide path to the CP txt file")

    args = parser.parse_args()
    DP = args.DP
    DPfile = open(DP, 'r')
    graph = defaultdict(lambda: defaultdict(lambda: ""))
    for line in DPfile.readlines():
        relation = line[:line.index("(")]
        head = line[line.index("(") + 1: line.index(",")]
        tail = line[line.index(",") + 2: line.index(")")]
        graph[head][relation] = tail

    tree = constituencyTree("ROOT-0")
    print(tree)
