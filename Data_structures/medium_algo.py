

# # Time O(n^2) || Space O(t) where t is no.of possible triplets
# def threeNumSum(array, targetSum):
#     '''
#     '''
#     result = []
#     sortedArray = sorted(array)
#     for index in range(len(sortedArray)-2):
#         leftPointer = index + 1
#         rightPointer = len(sortedArray) - 1
#         while leftPointer < rightPointer:
#             currentSum = sortedArray[index] + sortedArray[leftPointer] + sortedArray[rightPointer]

#             if currentSum == targetSum:
#                 result.append([sortedArray[index], sortedArray[leftPointer], sortedArray[rightPointer]])
                
#             if currentSum <= targetSum:
#                 leftPointer += 1
#             else:
#                 rightPointer -= 1
#     return result

# threeNumSum([12,3,1,2,-6,5,-8,6],0)
# =========================================================================================================

# # Time O(N + M) || Space O(1)
# def smallestDifference(array1, array2):
#     array1.sort()
#     array2.sort()
#     pointer1, pointer2 = 0, 0
#     result = [array1[pointer1] , array2[pointer2]]
#     min_diff = abs(array1[pointer1] - array2[pointer2])

#     while pointer1 < len(array1) and pointer2 < len(array2):
#         if abs(array1[pointer1] - array2[pointer2]) < min_diff:
#             min_diff = abs(array1[pointer1] - array2[pointer2])
#             result[0], result[1] = array1[pointer1] , array2[pointer2]
        
#         if array1[pointer1] < array2[pointer2]:
#             pointer1 += 1
#         elif array1[pointer1] > array2[pointer2]:
#             pointer2 += 1

#     print(min_diff)
#     return result

# print(smallestDifference([1, 3, 15, 11, 2],[23, 127, 235, 19, 8]))

#=======================================================================================

# # Time O(N) || Space O(1)
# def moveElementToEnd(array, ele):
#     '''
#     Takes in the array and the element.
#     Do move the instances of the given element
#     to the end of the array.
#     '''
#     beg = 0
#     end = len(array) - 1
#     while beg < end:
#         while beg < end and array[end] == ele:
#             end -= 1
#         if array[beg] == ele:
#             array[end], array[beg] = array[beg], array[end]
#         beg += 1

#     return array

# print(moveElementToEnd([2,1,2,2,2,3,4,2], 2))

#=======================================================================================


# class BST(object):
#     def __init__(self, node):
#         self.node = node
#         self.left = None
#         self.right = None
#     # Average case Time O(log(N)) || Space O(1)
#     # Worst case Time O(N) || Space O(1)
#     def insert(self, ele):
#         currentNode = self
#         while True:
#             if ele < currentNode.node:
#                 if currentNode.left is None:
#                     currentNode.left = BST(ele)
#                     break
#                 else:
#                     currentNode = currentNode.left
#             else:
#                 if currentNode.right is None:
#                     currentNode.right = BST(ele)
#                     break
#                 else:
#                     currentNode = currentNode.right
        

#     def search(self, ele):
#         currentNode = self
#         while currentNode is not None:
#             if ele == currentNode.node:
#                 return currentNode.node         
#             elif ele < currentNode.node:
#                 currentNode = currentNode.left             
#             else:
#                 currentNode = currentNode.right 
                
#     def delete(self, ele, rootnode):
#         #if rootnode == None:rootnode = self
#         if rootnode is None:
#             return None
#         if ele < rootnode.node:
#             rootnode.left = self.delete(ele, rootnode.left)
#         elif ele > rootnode.node:
#             rootnode.right = self.delete(ele, rootnode.right)
#         else:
#             if rootnode.left is None:
#                 rootnode = rootnode.right
#             elif rootnode.right is None:
#                 rootnode = rootnode.leftPointer
#             else:
#                 rootnode.node = self.findSuccessor(rootnode.right)
#                 self.delete(ele, rootnode.right)
#         return rootnode

#     def findSuccessor(self, parentnode):
#         while parentnode.left != None:
#             parentnode = parentnode.left
#         return parentnode.node

#     #O(N) Time || O(D) Space where D is depth of the bst
#     def validateBst(self, tree, minval=float("-inf"), maxval=float("inf")):
#         if tree is None:
#             return True
#         if tree.node < minval or tree.node > maxval:
#             return False

#         leftSubTree = self.validateBst(tree.left, minval, tree.node)
#         return leftSubTree and self.validateBst(tree.right, tree.node, maxval)


#     def preorder(self, rootnode):
#         print(rootnode.node)
#         if rootnode.left:
#             self.preorder(rootnode.left)
        
#         if rootnode.right:
#             self.preorder(rootnode.right)

#     def inorder(self, rootnode):
#         if rootnode.left:
#             self.inorder(rootnode.left)
#         print(rootnode.node)
#         if rootnode.right:
#             self.inorder(rootnode.right)

#     def longestBranch(self, rootnode):
#         _list = []
#         self.depthhelper(rootnode, _list, 0)
#         return _list

#     def depthhelper(self, rootnode, _list, length):
#         length += 1
#         if rootnode.left is None and rootnode.right is None:
#             _list.append(length)
#             return 
#         if rootnode.left:
#             self.depthhelper(rootnode.left, _list, length)
#         if rootnode.right:
#             self.depthhelper(rootnode.right, _list, length)
        
# bstobj = BST(10)
# bstobj.insert(5)
# bstobj.insert(15)
# bstobj.insert(3)
# bstobj.insert(2)
# bstobj.insert(22)
# bstobj.insert(13)
# bstobj.insert(34)
# bstobj.insert(36)
# bstobj.insert(100)
# print(bstobj.search(13))
# print('**************')
# bstobj.preorder(bstobj)
# print('**************')
# #bstobj.delete(13, bstobj)
# #bstobj.delete(13, bstobj)
# print('**************')
# bstobj.preorder(bstobj)
# print('**************')
# print(BST(2).validateBst(bstobj))
# print('**************')
# print(bstobj.longestBranch(bstobj))


#=======================================================================================

#Invert a binary tree:

# def invertBtree(tree):
#     queue = [tree]
#     while len(queue):
#         currentNode = queue.pop(0)
#         if currentNode is None:
#             continue
#         swapLeftAndRight(currentNode)
#         queue.append(currentNode.left)
#         queue.append(currentNode.right)

# def invertBtreeRec(tree):
#     if tree is None:
#         return
#     swapLeftAndRight(tree)
#     invertBtreeRec(tree.left)
#     invertBtreeRec(tree.right)


# def swapLeftAndRight(tree):
#     tree.right, tree.left = tree.left, tree.right
    
# **************************************************************

# Time O(N) | Space O(N)  ==> where N is the length of array
# def maxSumNotAdjacent(array):
#     maxsum = list()
#     for i in range(len(array)):
#         if len(maxsum) >= 2:
#             currentSum = max(maxsum[i-1], array[i] + maxsum[i-2])
#             maxsum.append(currentSum)
#         elif len(maxsum) == 1:
#             maxsum.append(max(array[i], array[i-1]))
#         else:
#             maxsum.append(array[i])
#     return maxsum.pop()

# def maxSumNotAdjacent(array):
#     runningSum = 0
#     runningSumPrev = 0
#     for i in range(len(array)):
#         if runningSum and runningSumPrev:
#             currentSum = max(runningSum, array[i] + runningSumPrev)
#             runningSumPrev = runningSum
#             runningSum = currentSum
#         elif runningSum and not runningSumPrev:
#             runningSumPrev = runningSum
#             runningSum = (max(array[i], array[i-1]))
#         else:
#             runningSum = array[i]
#     return runningSum

# result = maxSumNotAdjacent([7,10,12,7,9,14])
# result2 = maxSumNotAdjacent([8,9,3,7,1,24])
# print(result2)

#*************************************************************

# def waysCoinChange(targetSum, denominationArray):
#     '''
#     Do return the no.of ways to make the change of the given
#     targetSum from the denominations given in the array

#     :param1: targetSum
#     :param2: denominationArray

#     :return: no.of ways to make change
#     :rtype: int
#     '''
#     ways = [1] + [0]*(targetSum)
#     for denomination in denominationArray:
#         for amount in range(len(ways)):
#             if denomination <= amount:
#                 ways[amount] += ways[amount - denomination]
#     print(ways[targetSum])

# coinChange(10, [1,5,10,25])
# coinChange(9, [5])
#***********************************************************

# def minCoinChange(target, denoms):
#     minWays = [float("inf")] * (target + 1)
#     minWays[0] = 0
#     for denom in denoms:
#         for amount in range(len(minWays)):
#             if denom <= amount:
#                 remainingSum = amount - denom
#                 minWays[amount] =  min(minWays[amount] ,1 + minWays[remainingSum])
#         print(minWays)
    
#     return minWays[target]


# print(minCoinChange(6, [1,2,4]))

# ****************************************************************************

# def editDistance(str1, str2):
#     array = [ [ x for x in range(len(str1)+1)] for y in range(len(str2)+1) ]
#     for i in range(1, len(str2) + 1):
#         array[i][0] = array[i-1][0] + 1

#     for i in range(1, len(str2) + 1):
#         for j in range(1, len(str1) + 1):
#             if str2[i-1] == str1[j-1]:
#                 array[i][j] = array[i-1][j-1]
#             else:
#                 array[i][j] = 1 + min(array[i-1][j], array[i][j-1], array[i-1][j-1])
#     return array[len(str2)][len(str1)]

#______________________________________________

# def maxSumInSubset(array):
#     _maxSubSum = float('-inf')
#     _sum = 0
#     for i in range(len(array)):
#         _sum = max(_sum + array[i], array[i])
#         _maxSubSum = max(_sum, _maxSubSum)
#     return _maxSubSum

# print(maxSumInSubset([1,2,4,-10,1,2,6]))
# print(maxSumInSubset([3,5,-9,1,3,-2,3,4,7,2,-9,6,3,1,-5,4]))
# #[3,8,-1,1,4,2,5,9,16,18,9,15,18,19,]

#==========================================================


# def singleCycleCheck(array):
#     numVisits = 0
#     currentIdx = 0
#     while (numVisits < len(array)):
#         if numVisits > 0 and currentIdx == 0:
#             return False
#         numVisits += 1
#         currentIdx = getCurrentIdx(currentIdx, array)
        
#     #print(currentIdx)
#     return currentIdx == 0 

# def getCurrentIdx(currentIdx, array):
#     jump = array[currentIdx]
#     nextIdx = ( jump + currentIdx ) % len(array)
#     return nextIdx if nextIdx >= 0 else nextIdx + len(array)

# print(singleCycleCheck([2, 3, 1, -4, -4, 2]))
# ******************************************************************

# Time O(V+E) || Space O(V)
# class Node:
#     def __init__(self, value):
#         self.children = []
#         self.value = value

#     def insert(self, value):
#         self.children.append(Node(value))

#     def breadthFirstSearch(self, tree):
#         queue = [tree]
#         nodeValues = []
#         while len(queue):
#             currentNode = queue.pop(0)
#             nodeValues.append(currentNode.value)
#             for i in currentNode.child:
#                 queue.append(i)
#         return nodeValues

# ************************************************************************************

# River Sizes. 
# A two dimensional array with one's and zeros are given, and our algorithm has to determine 
# the largest connection of 1's in the array (those all 1's virtually form a river)

# def riverSizes(matrix):
#     sizes = []
#     visited = [ [ False for value in row] for row in matrix ]
#     for i in len(matrix):
#         for j in len(matrix[i]):
#             if visited[i][j]:
#                 continue
#             traverseNodes(i, j, matrix, visited, sizes)
#     return sizes

# def traverseNodes(i, j, matrix, visited, sizes):
#     riverSize = 0
#     nodesToExplore = [[i, j]]
#     while len(nodesToExplore):
#         currentNode = nodesToExplore.pop()
#         i, j = currentNode
#         if visited[i][j]:
#             continue
#         visited[i][j] = True
#         if matrix[i][j] == 0:
#             continue
#         riverSize += 1
#         unvisitedNeighbours = getUnvisitedNeighbors(i, j, matrix, visited)
#         for neighbour in unvisitedNeighbours:
#             nodesToExplore.append(neighbour)
#     if riverSize > 0:
#         sizes.append(riverSize)

# def getUnvisitedNeighbors(i, j, matrix, visited):
#     unvisited = []
#     if i > 0 and not visited[i-1][j]:
#         unvisited.append([i-1, j])
#     if i < len(matrix)-1 and not visited[i+1, j]:
#         unvisited.append([i+1, j])

#     if j > 0 and not visited[i][j-1]:
#         unvisited.append([i, j-1])
#     if j < len(matrix[0]) - 1 and not visited[i][j+1]:
#         unvisited.append([i, j+1])


#######################################################################################
# #Time :O(d) where d is deepest node
# #Space : O(1) 
#This is a tree  which supports .ancestor property. Get the youngest common ancestor from the given two #childs.
# def youngestCommonAncestor( child1, child2):
#     child1_depth = getDepth(child1)
#     child2_depth = getDepth(child2)
#     if child1_depth > child2_depth: 
#         #Do something
#     else:
#         #Do something

# def backtrackAncestor(lowerDescendant, higherDescendant, diff):
#     while diff > 0:
#         lowerDescendant = lowerDescendant.ancestor
#         diff -= 1
#     while lowerDescendant != higherDescendant:
#         lowerDescendant = lowerDescendant.ancestor
#         higherDescendant =higherDescendant.ancestor

#     return lowerDescendant


# def getDepth(child):
#     count = 0
#     while child.ancestor != None:
#         count += 1
#         child = child.ancestor
#     return count

#######################################################################################
# """
# Min binary heap is the heap data structure which have the min node value when compared to its
# respective child nodes
# """

# class MinHeap:

#     def __init__(self, array):
#         self.heap = array

#     def buildHeap(self):
#         firstParentIdx = (len(self.heap) - 2) // 2
#         for idx in reversed(range(firstParentIdx)):
#             self.siftDown(idx, len(self.heap)-1)
#         return self.heap

#     def siftDown(self, currentIdx, endIdx):
#         childOneIdx = currentIdx * 2 + 1
#         while childOneIdx <= endIdx:
#             childTwoIdx = currentIdx * 2 + 2 if currentIdx * 2 + 2 < endIdx else -1
#             if childTwoIdx != -1 and self.heap[childTwoIdx] < self.heap[childOneIdx]:
#                 idxToSwap = childTwoIdx
#             else:
#                 idxToSwap = childOneIdx
            
#             if self.heap[idxToSwap] < self.heap[currentIdx]:
#                 self.swap(idxToSwap, currentIdx)
#                 currentIdx = idxToSwap
#                 childOneIdx = currentIdx * 2 + 1
#             else:
#                 break


#     def siftUp(self, currentIdx):
#         parentIdx = (currentIdx - 1 ) // 2
#         while currentIdx > 0  and self.heap[parentIdx] > self.heap[currentIdx]:
#             self.swap(parentIdx, currentIdx)
#             currentIdx = parentIdx
#             parentIdx = (currentIdx - 1 ) // 2

#     def remove(self, value):
#         swap(0, len(self.heap) - 1)
#         valueToRemove = self.heap.pop()
#         self.siftDown(0, len(self.heap) - 1)
#         return valueToRemove

#     def insert(self, value):
#         self.heap.append(value)
#         self.siftUp(len(self.heap) - 1)

#     def swap(self, i, j):
#         self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

#     def peek(self):
#         return self.heap[0]

# minobj = MinHeap([102, 12, 21, 98, 55, 42, 29])
# print(minobj.heap)
# print(minobj.peek())
# print(minobj.buildHeap())

###################################################################################

# Time complexity - O(N) where N is total nodes in liked list
# Space complexity - O(1) 

# def removeKthNode(head, n):
#     '''
#     Removes kth node from the end in linked list.

#     Start with two pointers from the head to get the last kth node.
#     '''
#     firstPointer = head
#     secondPointer = head
#     while n:
#         secondPointer = secondPointer.next
#         n -= 1
#     if secondPointer is None:
#         head.value = head.next.value
#         head.next = head.next.next
#         return

#     while secondPointer.next != None:
#         firstPointer = firstPointer.next
#         secondPointer = secondPointer.next

#     firstPointer.next = firstPointer.next.next

#############################################################################################

# def getPermutations(array):
#     permutations = []
#     permutate(0, array, permutations)
#     return permutations

# def permutate(i, array, permutations):
#     if i == len(array) -1:
#         permutations.append(array[:])
#     else:
#         for j in range(i, len(array)):
#             swap(i, j , array)
#             permutate(i+1, array, permutations)
#             swap(i, j, array)

# def swap(i, j, array):
#     array[i], array[j] = array[j], array[i]

# print(getPermutations(['a', 'b', 'c']))
###################################################################################

# def makePermutations(array):
#     if len(array) == 0:
#         return []
#     elif len(array) == 1:
#         return [array]
#     else:
#         l = []
#         for i in range(len(array)):
#             iter_ele = array[i]
#             newarray = array[:i] + array[i+1 : ]
#             for x in makePermutations(newarray):
#                 l.append( [iter_ele] + x)
#         return l

# print(makePermutations(['a', 'b', 'c']))
#########################################################################

# Power set : A Power Set is a set of all the subsets of a set.

# def generatePowerSet(array):
#     powerset = [[]]
#     for ele in array:
#         for j in range(len(powerset)):
#             newset = powerset[j] + [ele]
#             powerset.append(newset)
#     return powerset
    
# result = generatePowerSet([1,2,3])
# print(result)

##################################################################################

def searchInSortedMatrix(matrix, target):
    '''
    Do search the given target number in the sorted matrix.
    matrix is sorted in both row and column.

    Time complexity : O(N+M) where N and M are rows and columns
    Space complexity : O(1)
    '''
    row = 0
    column = len(matrix[0]) - 1
    while row < len(matrix) and column >= 0:
        if matrix[row][column] == target:
            return [row, column]
        elif matrix[row][column] > target:
            column -= 1
        else:
            row += 1
    return [-1, -1] # when the target is not there in matrix
