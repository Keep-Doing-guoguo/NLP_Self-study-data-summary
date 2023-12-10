class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """

        # if n < 2:
        #     return 0 if n == 0 else 1
        if n < 2:

            if n == 0:
                return 0
            else:
                return 1
        m = self.fib(n-1) + self.fib(n-2)
        return m


print(Solution().fib(4))
'''
        n:
        4
        3
        2
        1 开始返回，返回值1

'''