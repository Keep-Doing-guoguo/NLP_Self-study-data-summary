class Solution(object):
    def numRescueBoats(self, people, limit):
        """
        :type people: List[int]
        :type limit: int
        :rtype: int
        """
        if people == None or len(people) == 0:
            return  0
        #sorted(people)#使用对撞指针，需要对数组进行排序。
        people.sort()
        i = 0
        j = len(people)-1
        res = 0
        while(i <= j):
            if people[i] + people[j] <= limit:
                i += 1
            j = j-1
            res = res +1
        return res
people = [3,2,2,1]
limit = 3
res = Solution().numRescueBoats(people,limit)
print(res)