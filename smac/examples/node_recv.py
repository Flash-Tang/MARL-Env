import socket
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='ip')
parser.add_argument('-p', dest='port', type=int, default=9990)

args = parser.parse_args()

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((args.ip, args.port))

f = open('message/msg.txt', 'a')
while True:
    data, addr = s.recvfrom(1024)
    f.write("%s: %s\n" % (addr, data))
    f.flush()
