import time
a = time.time()
counter = 0
while (counter < 1000000000):
  counter+=1

print(counter)

b = time.time()
print("use time:%d"%(b-a))