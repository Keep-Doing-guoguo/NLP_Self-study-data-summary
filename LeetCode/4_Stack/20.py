class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if len(s) == 0:
            return True
        stack = []
        for c in s:
            if c == '(' or c == '[' or c == '{':
                stack.append(c)
                #如果取完了，stack里面是空的，还是进到else里面，说明外面是多余的。
            else:
                if len(stack) == 0:
                    return False
                else:
                    temp = stack.pop()
                    if c == ')':
                        if temp != '(':
                            return False
                    if c == ']':
                        if temp != '[':
                            return False
                    if c == '}':
                        if temp != '{':
                            return False
        return True if len(stack) == 0 else False