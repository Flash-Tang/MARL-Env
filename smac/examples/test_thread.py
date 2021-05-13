import threading
import time


class MyThread(threading.Thread):
    def __init__(self):
        super(MyThread, self).__init__()

    def run(self):
        print(f'thread {threading.current_thread().name} begin')
        time.sleep(3)
        print(f'thread {threading.current_thread().name} terminates')


def main():
    thread_list = []

    for _ in range(1000):
        t = MyThread()
        thread_list.append(t)


    print(f'episode begins')
    start_time = time.time()
    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()
    print(f'episode ends')
    print(f'all threads run {time.time() - start_time}')


# def func():
#     print(f'thread {threading.current_thread().name} begin')
#     time.sleep(10)
#     print(f'thread {threading.current_thread().name} terminates')



if __name__ == '__main__':
    main()
