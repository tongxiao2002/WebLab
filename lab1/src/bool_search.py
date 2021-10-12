from enum import Enum

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

if __name__ == "__main__":
    test_case = 'NOT (("abdc" OR "bdef") AND ((NOT ("xt")) OR "xxxt"))'
    root = build_grammar_tree(test_case)
    print_tree(root)
