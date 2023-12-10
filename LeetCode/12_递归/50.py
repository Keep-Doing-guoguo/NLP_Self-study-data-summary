class Solution:
    def myPow(self,x,n):
        if n == 0:
            return 1.0
        y = self.myPow(n // 2)
        def quickMul(N):
            if N == 0:
                return 1.0
            y = quickMul(N // 2)
            return y * y if N%2 == 0 else 1
        pass