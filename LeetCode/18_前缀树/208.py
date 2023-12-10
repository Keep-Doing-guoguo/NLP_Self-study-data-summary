class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    def insert(self, word: str) -> None:
        node = self.root
        # 遍历单词中的每个字符
        for char in word:
            # 如果字符不在当前节点的子节点中，创建一个新节点
            if char not in node.children:
                node.children[char] = TrieNode()
            # 移动到子节点
            node = node.children[char]
        # 标记单词的最后一个字符节点为单词结尾
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        node = self.root
        # 遍历单词中的每个字符
        for char in word:
            # 如果字符不在当前节点的子节点中，说明单词不存在
            if char not in node.children:
                return False
            # 移动到子节点
            node = node.children[char]
        # 如果到达单词的最后一个字符节点，检查是否标记为单词结尾
        return node.is_end_of_word

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        # 遍历前缀中的每个字符
        for char in prefix:
            # 如果字符不在当前节点的子节点中，说明没有以该前缀开头的单词
            if char not in node.children:
                return False
            # 移动到子节点
            node = node.children[char]
        # 如果遍历完前缀中的所有字符，说明存在以该前缀开头的单词
        return True
trie =Trie()
trie.insert("apple")
trie.search("apple")  #// 返回 True
trie.search("app")     #// 返回 False
trie.startsWith("app") #// 返回 True
trie.insert("app")
trie.search("app")     #// 返回 True