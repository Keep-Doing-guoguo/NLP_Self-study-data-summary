# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head == None:
            return False

        slow = head
        fast = head
        while(fast!=None and fast.next!=None):
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                return True
        return False

num1 = ListNode(3)
num2 = ListNode(2)
num3 = ListNode(0)
num4 = ListNode(-4)
# num5 = ListNode(0)
# num6 = ListNode(0)
num1.next = num2
num2.next = num3
num3.next = num4
num4.next = num2
# num5.next = num6
Solution().hasCycle(num1)


