import time


a = 0
start_time = time.time()
for i in range(1000):
    a += 1
    print(a)
print(a)
print(f'run for {time.time() - start_time} seconds')
