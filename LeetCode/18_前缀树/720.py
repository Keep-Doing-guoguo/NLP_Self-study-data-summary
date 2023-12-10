class Trie:
    def __init__(self):
        self.children = {}
        self.isEnd = False
        self.val = ''


class Solution():
    def longestWord(self,words):
        if words == None or len(words) == 0:
            return ''
        root = Trie()

        for word in words:
            cur = root
            for c in word:
                if c in cur.children:
                    cur = cur.children[c]
                else:#在a的后面增加一个新的Trie，并且将cur指向到这个新的Trie
                    newNode = Trie()
                    cur.children[c] = newNode
                    cur = newNode
            cur.val = word
            cur.isEnd = True
        #Looking
        result = ''
        for word in words:
            cur = root
            if len(word) > len(result) or (len(word) == len(result) and word < result):
                isWord = True
                for c in word:
                    cur = cur.children[c]
                    if not cur.isEnd:
                        isWord = False
                        break
                result = word if isWord else result
        return result


words = ["a", "banana", "app", "appl", "ap", "apply", "apple"]
print(Solution().longestWord(words))