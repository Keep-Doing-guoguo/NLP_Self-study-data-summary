class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution():
    def reverseList(self,head):
        #第一个条件是空链表，第二个条件是递归结束的语句。
        if head == None or head.next == None:
            return head
        p = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return p

num1 = ListNode(1)
num2 = ListNode(2)
num3 = ListNode(3)
num4 = ListNode(4)
num5 = ListNode(5)
num1.next = num2
num2.next = num3
num3.next = num4
num4.next = num5
num5.next = None
Solution().reverseList(num1)