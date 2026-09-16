# Protocol check: is the inner packet split? (2026-09-14)

## A. Day-0 paired inner/outer capture (Cloudflare WARP, MASQUE h3-only)

| dir | inner n | outer n | matched 1:1 | overhead mode (B) | share at mode | inner >=1200 B | of which unmatched | inner max | outer max |
|---|---|---|---|---|---|---|---|---|---|
| up | 886 | 3258 | 757 (85.4%) | 43 | 98.3% | 27 | 0 | 1300 | 1353 |
| dn | 8335 | 7858 | 6040 (72.5%) | 43 | 95.0% | 280 | 12 | 1300 | 1359 |

Per inner protocol (whole IP packets of every family ride with the same offset):

| dir | inner proto | n | overhead = 43 B | unmatched |
|---|---|---|---|---|
| up | TCP4 | 292 | 233 | 53 |
| up | UDP4 | 594 | 511 | 76 |
| dn | IPv6 | 358 | 206 | 147 |
| dn | TCP4 | 391 | 346 | 42 |
| dn | UDP4 | 7586 | 5184 | 2106 |

Outer packets with no inner partner are the outer connection's own ACK-only packets: up: n=2501 top sizes [(46, 2301), (103, 43), (104, 22), (42, 19)], dn: n=1818 top sizes [(44, 765), (128, 11), (1088, 10), (1066, 10)]

## B. Outer-connection transport parameters (v1 raw pcaps, ClientHello in cleartext Initial)

| class | peer | max_udp_payload_size | max_datagram_frame_size | ClientHellos |
|---|---|---|---|---|
| Direct | ->origin | 1472 | 65536 | 625 |
| Masque | ->WARP | 1350 | 65536 | 520 |
| Masque | ->origin | 1472 | 65536 | 151 |
| Direct | ->WARP | 1472 | 65536 | 4 |

SNI of ClientHellos to WARP: {'www.quora.com': 4, 'consumer-masque.cloudflareclient.com': 520}

## C. Packet-size distribution: first-50-packet window vs full session (frame.len, bytes)

| view | class | dir | n | median | <120 B | >=1200 B | max |
|---|---|---|---|---|---|---|---|
| window_first50 | Direct | up | 4482 | 131 | 45.5% | 21.4% | 1292 |
| window_first50 | Direct | dn | 7218 | 1242 | 17.3% | 69.1% | 1505 |
| window_first50 | Masque | up | 7358 | 116 | 50.9% | 22.0% | 1365 |
| window_first50 | Masque | dn | 6142 | 116 | 53.5% | 10.8% | 1365 |
| full_session | Direct | up | 32140 | 87 | 72.9% | 12.9% | 1292 |
| full_session | Direct | dn | 119296 | 1292 | 4.6% | 90.4% | 1505 |
| full_session | Masque | up | 687310 | 146 | 26.5% | 2.7% | 1392 |
| full_session | Masque | dn | 1792843 | 1365 | 11.7% | 77.6% | 1392 |
