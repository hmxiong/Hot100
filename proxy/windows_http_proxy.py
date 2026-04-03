import asyncio
import argparse

async def relay(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except:
            pass

async def handle_client(reader, writer):
    try:
        req = await reader.readuntil(b"\r\n")
    except:
        writer.close()
        return
    line = req.rstrip(b"\r\n")
    parts = line.split(b" ")
    if len(parts) < 3:
        writer.close()
        return
    method = parts[0].upper()
    target = parts[1]
    headers = []
    while True:
        h = await reader.readuntil(b"\r\n")
        if h in (b"\r\n", b"\n"):
            break
        headers.append(h.rstrip(b"\r\n"))
    if method == b"CONNECT":
        if b":" in target:
            host, port = target.split(b":", 1)
            port = int(port)
        else:
            host = target
            port = 443
        try:
            r_reader, r_writer = await asyncio.open_connection(host.decode(), port)
        except:
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await writer.drain()
            writer.close()
            return
        writer.write(b"HTTP/1.1 200 Connection established\r\nProxy-Agent: PyProxy\r\n\r\n")
        await writer.drain()
        await asyncio.gather(relay(reader, r_writer), relay(r_reader, writer))
        return
    host = None
    for h in headers:
        if h.lower().startswith(b"host:"):
            host = h.split(b":", 1)[1].strip()
            break
    if host is None:
        writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    if b":" in host:
        h, p = host.split(b":", 1)
        port = int(p)
        host_name = h.decode()
    else:
        host_name = host.decode()
        port = 80
    path = target.decode()
    if path.startswith("http://") or path.startswith("https://"):
        idx = path.find("/", path.find("://") + 3)
        if idx != -1:
            path = path[idx:]
        else:
            path = "/"
    try:
        r_reader, r_writer = await asyncio.open_connection(host_name, port)
    except:
        writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    r_writer.write(method + b" " + path.encode() + b" HTTP/1.1\r\n")
    for h in headers:
        r_writer.write(h + b"\r\n")
    r_writer.write(b"\r\n")
    await r_writer.drain()
    await asyncio.gather(relay(reader, r_writer), relay(r_reader, writer))

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8888)
    args = ap.parse_args()
    srv = await asyncio.start_server(handle_client, args.bind, args.port)
    async with srv:
        await srv.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
