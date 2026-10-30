import socket
for p in (9527, 9528, 9529, 9530):
    s = socket.socket()
    s.settimeout(1)
    r = s.connect_ex(('localhost', p))
    s.close()
    print(p, 'OPEN' if r == 0 else 'CLOSED')
