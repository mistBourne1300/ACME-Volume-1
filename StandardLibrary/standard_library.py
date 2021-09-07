# standard_library.py
"""Python Essentials: The Standard Library.
<Name>
<Class>
<Date>
"""
import calculator as calc
import itertools as itrt
import sys, time
import random as rand
import os #will use if I have time to implement shut the box better



# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """
    return min(L), max(L), sum(L)/len(L)


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    int1 = 0
    int2 = int1
    str1 = "hello"
    str2 = str1
    list1 = [0,1,2,3,4,5]
    list2 = list1
    tuple1 = (0,0)
    tuple2 = tuple1
    set1 = {"hello", 'yes', 'no', 'please', 'thank you'}
    set2 = set1
    int1 += 2
    str2 += "world"
    list2[0] = 1000
    tuple2 = (1,1)
    set2.add("hello world")
    print("ints mutable: ", int1==int2)
    print("strings mutable: ", str1==str2)
    print("lists mutable: ", list1==list2)
    print("tuples mutable: ", tuple1==tuple2)
    print("sets mutable: ", set1==set2)


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt that are 
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    return calc.sqrt(calc.sum(calc.product(a,a),calc.product(b,b)))


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    #print(list(itrt.combinations(A,1)), end= "\n\n\n")
    powerSet = []

    for i in range(len(A)+1):
        powerSet += set(itrt.combinations(A,i))
    powerSet = list(itrt.chain(powerSet))

    return powerSet


# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""

    print("Welcome ", player, "!\nReady to begin? (press enter)")
    input("")
    print("Time Limit: ", timelimit, " sec")
    numbers = [i for i in range(1,10)]
    print(numbers)
    startTime = time.time()
    endTime = startTime + int(timelimit)
    die1, die2 = 0, 0
    playerWin = False
    while(time.time() < endTime) and (len(numbers)>0):
        if sum(numbers)>6:
            die1, die2 = rand.choice(range(1,7)), rand.choice(range(1,7))
            dieSum = die1 + die2
        else:
            dieSum = rand.choice(range(1,7))


        print("Numbers left: ", numbers)
        print("Roll: ", dieSum)
        print("Seconds left: ", round(endTime - time.time()))
        possibleCombos = power_set(numbers)
        able_to_play = False
        for i in range(len(possibleCombos)):
            if sum(possibleCombos[i]) == dieSum:
                able_to_play = True
                break
        if not able_to_play: break
        numberChoices = []
        correctInput = False
        while not correctInput:
            nums_to_remove_str = input("Enter the numbers to remove (Must be single-digit numbers separated by spaces: ")
            nums_to_remove = list(nums_to_remove_str[0::2])
            print(nums_to_remove)
            numberChoices = ([int(num) for num in nums_to_remove])
            correctInput = True
            for num in numberChoices:
                if  num not in numbers:
                    correctInput = False
                    print("problem: ", num)
                    print("Incorrect input, try again")
                    print("Seconds left: ", round(endTime - time.time()))
                    break
        


        
        print(numberChoices)
        if (time.time() > endTime): 
            if len(numbers) == 0:
                playerWin = True
            break

        if(sum(numberChoices) == dieSum):
            numbers = [i for i in numbers if i not in numberChoices]
            print(numbers)
        else:
            print("Cannot remove numbers, sum not equal to die sum")
    
    if playerWin:
        print("congratulations, you won!")
    else:
        print("Good, try, but no dice")
    
    print("Score for", player, ":", sum(numbers))
    print("Time Played: ", time.time() - startTime)
        
        










if __name__ == "__main__":
    prob1([0,1,2,3,4,5,6])
    prob2()
    print("hypotenuse: ", hypot(3,4))
    print(power_set([]))
    if len(sys.argv) != 3:
        print("Exactly two extra command line arguments required for shut the box")
        print("system commands: ", sys.argv)
    else:
        shut_the_box(sys.argv[1], sys.argv[2])