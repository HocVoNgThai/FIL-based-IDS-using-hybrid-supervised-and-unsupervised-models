ip = "1.1.1.1"
import ipaddress

ret = float(int(ipaddress.IPv4Address(ip))) / 1e9
print(ret)