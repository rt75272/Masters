from random import random


heads_wins = 0
tails_wins = 0
n = 1000000
for i in range(n):
    x = random() * 10
    # print(x)
    if(x < 5):
        print("heads")
        heads_wins+=1
    elif(x > 5):
        print("tails")
        tails_wins+=1

heads_ratio = heads_wins / n * 100
tails_ratio = tails_wins / n * 100

heads_ratio = round(heads_ratio, 4)
tails_ratio = round(tails_ratio, 4)


print("Heads win ratio: " + str(heads_ratio))
print("Tails win ratio: " + str(tails_ratio))