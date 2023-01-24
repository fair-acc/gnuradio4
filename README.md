[![License](https://img.shields.io/badge/License-LGPL%203.0-blue.svg)](https://opensource.org/licenses/LGPL-3.0)
![CMake](https://github.com/fair-acc/graph-prototype/workflows/CMake/badge.svg)
# Graph Prototype
A small proof-of-concept for evaluating efficient [directed graph](https://en.wikipedia.org/wiki/Directed_graph)-based
algorithms, notably required node structures, scheduling interfaces, and partial compile-time merging of
[directed acyclic](https://en.wikipedia.org/wiki/Directed_acyclic_graph) as well as 
[cyclic graphs](https://en.wikipedia.org/wiki/Feedback_arc_set) (aka. feedback loops).  

The expressed goal is to guide the low-level API design and functionality for the upcoming 
[GNU Radio 4.0](https://github.com/gnuradio/gnuradio/tree/dev-4.0) release.

A [single header version](https://raw.githubusercontent.com/fair-acc/graph-prototype/single-header/singleheader/graph.hpp) is provided in the `dist/`subdirectory, which can be regenerated with the merge_headers.sh script.
It can be used on [compiler-explorer](https://godbolt.org/) with `-sdt=c++20 -O3` compiler options: [compiler-explorer example](https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQWm8GA6ZAB36kqqBgQDKyJvRABGACzDRBAKoBnTAAUAHiADkABj2kAViABMpWkwahkAUnMAhR09Ib6yAnlGVG6AGFUWgBXFgY5Un8AGW5MADkwgCNMYjkANkt%2BVDVCHwYg0PDI7NzvUViGBOTUjMsPTC988QImYgJCsIjZd0xPcoYWtoJK6pYUtNlM91b2zuKetVmRuMTx2qnzAEp3VBDiZHY9RwBmbmRQrABqexOAtuImAE9b7HsDAEFT88vMG7uqCwCAB6NDETA8BCvd5fT7fBgXELXRzmBAEAj8NQgYHAx4Adx4wEICBCSRCGgOykYBD4qBYwKoXGIAFomMhkMDgI9%2BAhmfxiKgiAQnvxMMDcrZ6MyEJgmFhiOLuMB6DK5alOdyEJDBCiYTDmGw1Pw2X8qMB/gARK6MvBpEBcpg824uOGfAiYFj8azu24BYWig1/AAq0M%2BFyYajUV0eDHQdIA%2Bmo9gc/iArvxSbQ8MhraAQAwMJhfTG4yxE8nDr6QydsKRcyAQAB5FTVgJBusokt03U1143ADsLo%2BGaSWeQIBhVynVzQDCWmG0/KuIc%2B0/TAsOkfjokwEC2A6Ha7X4II%2BwYV3k5mdk%2Bn9n7Fr19%2Bvro%2B7s93qLd39jFYwdDH3DSMZz2AhE24ABrK40xHMd63zQtfTQEJQIlcCq1eOszQbABJeIqw7cxzFQns3hrA8YRg7MJ1XacADdUDwdAbynflUE3NRtyqCAgyuWjpBCTB9zvQ8jynJZ0AbJCCH%2BAJfV4/i/l9OTxIbfxaGfD41zvB84SfE4hxhN8vSYH0vxFH82GXOseMkaRMDrb9AyuAAlS0riwC5vwgFSQA82g%2BPUu5q2wPcrgAKiuHy/IC9Caz3LZ/0AqM1CkegoPTTNszggssF9FK7Pwq5bPoOtnIwuDcMKlESDwIlmHUwjyqwpsW19ZyCKI1LMCYxrezI4SKMy8dmKuIyP19G07TYVoG1AkgwJYJigr7AA1f8tIAVhcLacrwfLiCYjaHyO4C53dRdiCuJhkNQEbWPYzjdxWq791nJYrgLBdDn4aSBpo0STzPK7wqKrqNK0p9dIffS9TdD1jNMv1zKc9tRuR38XLcvyvKixp/OkWKQqE5xIoICTfLxmLlrirYEt7MNrCAuV5XSyic2anLP3udB5UJzC80qoKOuZ/wDBI/mcLwoWbkIkXY1kcW4ObVt2ploiwhIvs/uHIbqM06cxpMrnJobaamFm7diAWpa21W9bby2%2BwdowPapAOp3jqtN7zqXa6iDujdMC3HcIGepg62epJXtEd7Pu0b7fsHEbj0wU9iHPJgZacK4knB29Ia%2BPSDLh98jd9RyMdRnzcgAL0weNpKCEIxDcq96YAxmo3QEIvWzI3WaG7KELubve6kRHUabsQmqZU3U/NkBvyzJYJoFqW2w67hNZrCWQHBUUjfQeNsnaNRfSnghrI6kCUV3k/QO/BuG3EINnOwD4AFk75IUD8HBJpRBPxakGTQrU%2BrYHIv9K45IlQ5wjIpE4VoObDwCKPMcZdpYXxnraOeM1F7mWXgQVeksqqES3r1UitY4L71lO6I%2B98z53AvlfNWN9CLf3aA3cyQCX5v0/hw3%2BtpGgDCAcrUBwVoQwxfGzPWa4YG2GjKnM8XDRRuQrpZJI8CGyA3Tioz8xd9ZTidttJwu19qHU9qdecF1FFpwYHogObEg4cRDjxJg0czofVQF9TAP1IGGKPDo88xjHDpCYB7PKZMGy13rjSHgPArjYUYSFHy3AsDaETJgAAjgJBEXMknxJ4K8ISg5bFAx8pwcCsSe70CgBAeijF9xJLrO4wpdNs7aWTtObyUSOBMCqfGNJC5Mk5MYJWJhexp403aSNTpUMNKGXhuNMyAZK51mrngOuDcrgWl8QQKE7cU45KEWoCAuyfoIH%2BBAgwCUGYRi7n0Z4A9RxZWQblEejyXjS3Ofspqa9SHmHIW3KhzVlYArYcCrWScoE%2BQeM8QqPyDkQLJFQKgqQ3IDSLiNbg0lRJ4rXKULSiCrhiykbCHWLzhpQJCc7fA5iIknW9guJcK4An3WcY9biVxuD7jjgnfx%2BKeInjxbcK0KK0XEGMaUCJZL8XitSFKnIHs3Jb1lXivAVArgQEJaK0VOy9mXOZFcWQJSRL4pYjkNypKzVHm0lcPoGgBXmqMc4Vw0q1WiTmQElOdjbF5yMQXTpZKYQ4phJwbgoVtZyIlOaZqbBiDAH0SNeRsa8zxsTfGJITxBmxgXAs6l0LvXThxO5PGbRTS0FQHiZkDoeRpnMFcbQ3LzzMj7CLS6rayL5SlFm5kDbO0QO7ZgZkvaWDcHJLIK4A6rggQzEQqBa4/aoCuOm7qblV2ZuzUM7QvoDB1lJSksGdwcV1mZArGmdYN1Zpzek3d%2B7XjeSPQEE9VxgV7haTzBVx6plE1pvmotU4S0LkaMhP4taoQLunLC4gjwvnPrEHWeQbaMWlJ6K%2BusJxEMDmhjatc0HYO%2BhfUhsiSQUPZ1kHu41lGKN1go9h/9%2BKcXOoBla/1R4RCXR6eTGJ2y8Csf0tymSF5nRZ1cHgU1XSWOuD1au9APB2XBy4uEraeAPZ1lzipj2MzIMBp0uS/FgICANn5DiiAKJnJBxCLQaSqBNXgftdoEDAw0yYuOgEBgt9ozaf03i%2B5qQCAQEurq4l5gblsa9bMwt%2BKl0rtSIm9A664v12vduu9JKH1oL7ojF9b6diRSfTlh9f7g06auEBxzyBQNXHA5J0m5M4VwaI8hvVwljV1ksFcTDF56MlYA9aEgWqNlbOknxvV1rBNyWIx011zhxNOudTF08Xo/i3mJbJ%2BTgdFO7mU04VTR1vPMaujdG4W1iBoeIFeE6eqlv0DYwZoExniCmd1Ene85E3tO3c55s7dYLsHZFQXCGenIu4enDF2TiWE1cxREOnqHWURy3QArChj6Crfsvq%2BorH7ebo6K952r5WnNgc1LV/D8L0eIea8S1raGOtdfkD10HYlekNcIwh4TJGyNUdo9R3njPYZ9aY4d6cQXiXWtqxxwbvSeMjf49nUbdwOfTZcLNiTpXjyibW0luTCmXFKeMXti06mDdabu/nYHpXDOPee4RCzagrM2bs5qBzTn8gude25jz7CvMMd85GfzgXLR6vSGFj19GXwQyZ0dogsWocJZk0lzdN6813DQweiAV6t25p3XcSj6fMvjy5k1i9%2BW0fwYx7lvLQ62cV7x770ShPKvumqyT0rZPGvs%2BIxAzOLXUPtYw1hoNBiDMDa49EzZsSJti5EwrgISvROq/m%2BaxbNSVvOoT3HjbTitsQB24b/7C3jvGJ%2B9GDrxAuvECQ1d4lN2k3q%2BnFbkAJmxBmcah797VpXMf5O1973J%2BLu/YX7yAH62qA7m4C6R7R7LpYDWBPBrp6owHk7l7tbQge4iYRYFpR4xZdgsBuQ4Hlj7DjLl6oEuBYoR6%2BbHYQ4b4ZopbZ5pbp4VKoC0S7g4F5ZSRgQMBoS47TJm5Tjra66cogEYEBJRoUEx44F4E2ClgEEpg14kFD4QFiHLpUFa5Q5KSEQ4GebVRPZ1QEwo6MHMGBZSF0hV4Fbs6V68Gx7xZb4PQhxCGBoFx6A7C0D6AbTGARCGDGCoB9LcB8CCCkBGaeG0ykDgQgAbR7ouF6DyDuH6BGCkDeF6DGBYh7qBFGA7BwCwAwCIAoB0j8B0CpDkCUBoCej5FpDACyDsJYC0TZiYArR4CYB4iNgBj6CyDGA0DWapBYgQBJCxHGBJDcBtBPAtF9EDHEBPCNhJAmDCLDGkDFFsBiCNgMC0BDGeGkBYCcCShHBxF/zCJ4DMFYirHAZN5bHGA4p9C9FWB4BJCPBjFBBYAXEEBPYsDDE7BUDWDABqB1ENFNGMAzEiBiDFTsAKBKBiDqBaC6CrFmA9DWC2AoDSbOCXFJBYiwCBggCGakDMFpB6AAAcsgrRwRqAP0%2BQBxzI4koqDgM2TgoWU6jYJwSRfQuxvgEA/g8w3QUQsYowawEwJQOQeQogrJPJZQ%2BQnJNQkwvQ/QzQywApiwDJACgwywIp6wYpSwww0pMwwwip3JJquwhB7A2wVgrhMRqxCRja2J6QzI6QDOwA7Ixq5gPADaEAuAhAA2pw2wxgqRThOwqo8olAOwoR4RBpURxgzxAZHhcRCRSRIAKRvR6RWREASAs6yEhREAxReR9AxA8Qv4%2Bg2gZpFpVpNpFR9pxg3UzpT2Eklg/xEgXUcgiglZYJOgFxUJpAeI3ILxgZbhpAYZXh%2BgjYyEc6M6mqOZ5plp1WBZdpDpQQJR6ZMsJwJq7pMZXpsoPp76kR0RpAIZe6XZ8R%2BgkZ0ZQRfpYRER%2BgdJnZFxEZARC5gZ5gRp4ZO5F5%2B5GJnRbu8gQAA%3D)

## Copyright & License
Copyright (C) 2018-2022 FAIR -- Facility for Antiproton & Ion Research, Darmstadt, Germany<br/>
Unless otherwise noted: [SPDX-License-Identifier: LGPL-3.0-or-later](https://spdx.org/licenses/LGPL-3.0-or-later.html)

### Contributors
 * Ivan Čukić <ivan.cukic@kdab.com>
 * Matthias Kretz <M.Kretz@GSI.de>
 * Alexander Krimm, <A.Krimm@GSI.de> 
 * Ralph J. Steinhagen, <R.Steinhagen@GSI.de>

### Acknowledgements
...

