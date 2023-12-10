# class Solution(object):
#     def nextGreaterElement(self, nums1, nums2):
#         """
#         :type nums1: List[int]
#         :type nums2: List[int]
#         :rtype: List[int]
#         """
#
#         stack = []
#         ht = {}
#         res = []
#         for num in nums2:
#             #栈不为空，且num始终还得大于栈顶元素，才可以取出来，放入到table里面
#             while len(stack) != 0 and num>stack[-1]:
#                 temp = stack.pop()
#                 ht[temp] = num
#             stack.append(num)
#
#         while len(stack) !=0 :
#             ht[stack.pop()] = -1
#         for num in nums1:
#             res.append(ht[num])
#         return res
#


class Solution(object):
    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """

        res = []
        stack = []
        for num in nums2:
            stack.append(num)

        for num in nums1:
            temp = []
            isFound = False
            max = -1
            while (len(stack) != 0) and (not isFound):
                top = stack.pop()
                if top > num:
                    max = top
                if top == num:
                    isFound = True#这个是用来跳出循环使用的
                temp.append(top)
            res.append(max)
            while len(temp) != 0:
                stack.append(temp.pop())
        return res


nums1 = [4,1,2]
nums2 = [1,3,4,2]
Solution().nextGreaterElement(nums1,nums2)