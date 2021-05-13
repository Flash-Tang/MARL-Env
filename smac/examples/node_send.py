import socket
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i1', dest='ip1')
parser.add_argument('-i2', dest='ip2')
parser.add_argument('-p', dest='port', type=int, default=9990)
parser.add_argument('-m', dest='message', default=b'hello')

args = parser.parse_args()

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.sendto(args.message, (args.ip1, args.port))
s.sendto(args.message, (args.ip2, args.port + 1))
s.close()
