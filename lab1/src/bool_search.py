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
    def __init__(self, type=TokenType.NONE, value="") -> None:
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
        elif text[idx:idx + 3] == "NOT":
            lex_result.append(Token(TokenType.NOT, "NOT"))
            idx += 3
            continue
        elif text[idx:idx + 3] == "AND":
            lex_result.append(Token(TokenType.AND, "AND"))
            idx += 3
            continue
        elif text[idx:idx + 2] == "OR":
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


def print_tree(root: Tree):     #
    if len(root.children) == 0:
        print(root.token.value, end=" \n")
    elif len(root.children) == 1:
        print_tree(root.children[0])
        print(root.token.value, end=" \n")      
    elif len(root.children) == 2:
        print_tree(root.children[0])        
        print_tree(root.children[1])
        print(root.token.value, end=" \n")
    else:
        raise ValueError

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

def tree_search(indicesfile:str):
    #text = input()
    #text = 'NOT (("abdc" OR "bdef") AND ((NOT ("xt")) OR "xxxt"))'
    text = '"card" OR "member"'
    root = build_grammar_tree(text)
    
    pickle_file = open(indicesfile, 'rb')
    inverse_indices = pickle.load(pickle_file)

    id_list = list(inverse_indices["UUID"].keys())  #for not option

    stack = tree_to_stack(root)
    fullset = set(inverse_indices.keys())
    setstack = []
    porter_stemmer = nltk.stem.PorterStemmer()

    for node in stack:
        if node.token.type == TokenType.WORD:     #TokenType.WORD:
            word = porter_stemmer.stem(node.token.value)
            if word in inverse_indices.keys():
                set1 = set(inverse_indices[word].keys())
            else:
                set1 = set()
            setstack.append(set1)
        elif node.token.type == TokenType.NOT:     #:
            setstack[-1] = fullset.difference(setstack[-1])
        elif node.token.type == TokenType.AND:       #:
            setstack[-2] = setstack[-2] & setstack[-1] 
            setstack.pop()
        elif node.token.type == TokenType.OR:       #:
            setstack[-2] = setstack[-2] | setstack[-1] 
            setstack.pop()
        #print(len(setstack))

    return setstack[0]

if __name__ == "__main__":
    #test_case = 'NOT (("abdc" OR "bdef") AND ((NOT ("xt")) OR "xxxt"))'
    indicesfile = "D://WorkPlace/data/output/invert_indices.dict"
    #root = build_grammar_tree(test_case)
    #print_tree(root)
    #stack = tree_to_stack(root)
    #print(stack) 
    set = tree_search(indicesfile)
    print(set)

