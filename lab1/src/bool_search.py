from enum import Enum
import pickle
import nltk


class TokenType(Enum):
    NONE = 0
    WORD = 1
    NOT = 2
    AND = 3
    OR = 4
    LB = 5      # left round brackets
    RB = 6      # right round brackets


class Token():
    def __init__(self, type=TokenType.NONE, value=""):
        '''
        :param type         Token 种类
        :param value        Token 的值，字符串
        '''
        self.type = type
        self.value = value


class Tree():   
    def __init__(self, token: Token):
        self.token = token
        self.children = []
    
    def add_children(self, token: Token):
        self.children.append(token)


def lexer(text: str):
    lex_result = []
    idx = 0
    while idx < len(text):
        if text[idx] == " ":
            idx += 1
            continue
        elif text[idx] == "(":
            lex_result.append(Token(TokenType.LB, "("))
            idx += 1
            continue
        elif text[idx] == ")":
            lex_result.append(Token(TokenType.RB, ")"))
            idx += 1
            continue
        elif text[idx] == "\"" or text[idx] == "'":
            next_quota_idx = text[idx + 1:].find(text[idx]) + idx + 1
            lex_result.append(Token(TokenType.WORD, text[idx + 1:next_quota_idx]))
            idx = next_quota_idx + 1
            continue
        elif text[idx:idx + 3].upper() == "NOT":
            lex_result.append(Token(TokenType.NOT, "NOT"))
            idx += 3
            continue
        elif text[idx:idx + 3].upper() == "AND":
            lex_result.append(Token(TokenType.AND, "AND"))
            idx += 3
            continue
        elif text[idx:idx + 2].upper() == "OR":
            lex_result.append(Token(TokenType.OR, "OR"))
            idx += 2
            continue
        else:
            raise ValueError("Illegal string.")
    return lex_result


def parser(lex_result: list):
    def token_pair_strip(lex_result: list, ltokentype: TokenType, rtokentype: TokenType):
        while lex_result[0].type == ltokentype and lex_result[-1].type == rtokentype:
            depth = 0
            for idx, token in enumerate(lex_result[1:-1]):
                if token.type == TokenType.LB:
                    depth += 1
                    continue
                elif token.type == TokenType.RB:
                    depth -= 1
                    continue
                if depth < 0:
                    # indicates that lex_result[0] and lex_result[-1] is not paired, refuse to delete.
                    return lex_result
            lex_result = lex_result[1:-1]
        return lex_result

    lex_result = token_pair_strip(lex_result, ltokentype=TokenType.LB, rtokentype=TokenType.RB)
    root = Tree(Token())
    depth = 0
    for idx, token in enumerate(lex_result):
        assert depth >= 0, "Illegal bracket pairs."
        if token.type == TokenType.LB:
            depth += 1
            continue
        elif token.type == TokenType.RB:
            depth -= 1
            continue
        elif (token.type == TokenType.AND or token.type == TokenType.OR) and depth == 0:
            root = Tree(token)
            root.add_children(parser(lex_result[:idx]))
            root.add_children(parser(lex_result[idx + 1:]))
            break
    assert depth == 0, "Illegal bracket pairs."
    if root.token.type == TokenType.NONE:
        # this indicates that there doesn't exist top AND and OR in lex_result
        if lex_result[0].type == TokenType.NOT:
            # NOT clause
            root = Tree(lex_result[0])
            root.add_children(parser(lex_result[1:]))
        else:
            # single word
            root = Tree(lex_result[0])
    return root


def build_grammar_tree(text: str):
    lex_result = lexer(text)
    root = parser(lex_result)
    return root


def print_tree(root: Tree):
    if len(root.children) == 0:
        print(root.token.value, end=" ")
    elif len(root.children) == 1:
        print(root.token.value, end=" ")
        print_tree(root.children[0])
    elif len(root.children) == 2:
        print_tree(root.children[0])
        print(root.token.value, end=" ")
        print_tree(root.children[1])
    else:
        raise ValueError


def compute_tree(invert_indices: dict, root: Tree) -> set:
    if root.token.type == TokenType.WORD:
        token = root.token.value
        try:
            doc_id = set(invert_indices[token].keys())
        except KeyError:
            raise KeyError("token '{}' doesn't exist!".format(token))
        return doc_id
    elif root.token.type == TokenType.NOT:
        N = 306242
        token = root.token.value
        total_doc = set([idx for idx in range(N)])
        return total_doc.difference(set(invert_indices[token].keys()))
    elif root.token.type == TokenType.AND:
        left_result = compute_tree(invert_indices, root.children[0])
        right_result = compute_tree(invert_indices, root.children[1])
        return left_result.intersection(right_result)
    elif root.token.type == TokenType.OR:
        left_result = compute_tree(invert_indices, root.children[0])
        right_result = compute_tree(invert_indices, root.children[1])
        return left_result.union(right_result)
    else:
        raise ValueError("Illegal Operator.")


def bool_search(invert_indices: dict, expr: str) -> set:
    root = build_grammar_tree(expr)
    return compute_tree(invert_indices, root)


def tree_to_stack(root: Tree):
    if root == None:
        return False
    stack1 = []
    stack2 = []
    stack1.append(root)
    while stack1:
        node = stack1.pop()
        if(len(node.children)) == 1:
            stack1.append(node.children[0])
        elif (len(node.children)) == 2:
            stack1.append(node.children[0])
            stack1.append(node.children[1])           
        stack2.append(node)
      
    stack2.reverse()
    return stack2


def bool_search(indicesfile:str, boolsearchfile:str):
    #text = 'NOT (("abdc" OR "bdef") AND ((NOT ("xt")) OR "xxxt"))'
    text = '(("company" or "precent") ANd ((NOT ("income")) OR "march"))'
    root = build_grammar_tree(text)
    
    pickle_file = open(indicesfile, 'rb')
    inverse_indices = pickle.load(pickle_file)

    stack = tree_to_stack(root)
    fullset = set(uuid_indice.keys())
    setstack = []
    porter_stemmer = nltk.stem.PorterStemmer()

    for node in stack:
        if node.token.type == TokenType.WORD:    
            word = porter_stemmer.stem(node.token.value)
            if word in inverse_indices.keys():
                set1 = set(inverse_indices[word].keys())
            else:
                set1 = set()
            setstack.append(set1)
        elif node.token.type == TokenType.NOT:     
            setstack[-1] = fullset.difference(setstack[-1])
        elif node.token.type == TokenType.AND:       
            setstack[-2] = setstack[-2] & setstack[-1] 
            setstack.pop()
        elif node.token.type == TokenType.OR:      
            setstack[-2] = setstack[-2] | setstack[-1] 
            setstack.pop()
    return setstack[0]


if __name__ == "__main__":
    indicesfile = "lab1/data/output/invert_indices.dict"
    boolsearchfile = "lab1/data/boolsewsarchwords.txt"
    id2uuidfile = "lab1/data/output/id2uuid.dict"

    idset = bool_search(indicesfile, boolsearchfile)

    pickle_idfile = open(id2uuidfile, 'rb')     #id to uuid 
    uuid_indice = pickle.load(pickle_idfile) 
    for id in idset:
        if id in uuid_indice:
            print(uuid_indice[id])
        else:
            print("illegal id number")

    

