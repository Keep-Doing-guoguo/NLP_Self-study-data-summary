class Solution():
    #1.使用双指针解法，来进行解决。
    #=====================
    def reverseString(self,s):
        l ,r = 0 , len(s) - 1
        while(l <= r):
            s[l],s[r] = s[r],s[l]
            l = l + 1
            r = r - 1

    # =====================


    #2.使用递归的思想
    # =====================
    def reverseString_recursion(self,s):
        self.recurion(s,0,len(s)-1)
    def recurion(self,s,l,r):
        if l >= r:
            return
        self.recurion(s,l+1,r-1)
        s[l],s[r] = s[r],s[l]
    # =====================
s = ["h","e","l","l","o","H","a","n","n","a","h"]
Solution().reverseString_recursion(s)