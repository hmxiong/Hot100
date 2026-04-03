import os
import sys
import argparse
import subprocess
from pathlib import Path

def set_env(proxy):
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy

def write_condarc(proxy):
    home = Path.home()
    condarc = home / ".condarc"
    content = "proxy_servers:\n  http: {0}\n  https: {0}\n".format(proxy)
    condarc.write_text(content, encoding="utf-8")

def test_https(proxy):
    import socket
    host = "www.baidu.com"
    port = 443
    s = socket.socket()
    s.settimeout(5)
    s.connect((host, port))
    s.close()
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", required=True)
    ap.add_argument("--env", default="")
    ap.add_argument("--spec", default="python=3.10")
    args = ap.parse_args()
    set_env(args.proxy)
    write_condarc(args.proxy)
    try:
        test_https(args.proxy)
        print("ok")
    except Exception as e:
        print("fail")
    if args.env:
        subprocess.run(["conda", "create", "-n", args.env, args.spec, "-y"], check=False)

if __name__ == "__main__":
    main()
