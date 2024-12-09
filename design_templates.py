import numpy as np

m2 = np.array(
    [
        [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0],
    ]
)

urls = [
    {
        "url": "https://d.newsweek.com/en/full/2463914/haliey-welch.jpg?w=1200&f=daddd6871e66c2facf472a455c0a94d6",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://www.rollingstone.com/wp-content/uploads/2024/12/hailey-welch-hawk-memecoin.jpg?w=1581&h=1054&crop=1",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://content.api.news/v3/images/bin/7a30da19649e6f81aac060bdaa9f59c3",
        "method": "alternative",
        "parameters": (1.05, 3),
        "quantity": 1,
    },
    {
        "url": "https://www.pajiba.com/assets_c/2024/12/hawktuahscam-thumb-700xauto-267027.png",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://i.ytimg.com/vi/JKGT6T4R3Pg/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLBJZxg3SVov1N9K4Dm0H2AHpCehHw",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://townsquare.media/site/204/files/2024/07/attachment-HawkTuah.jpg",
        "method": "alternative",
        "parameters": (1.1, 3),
        "quantity": 2,
    },
    {
        "url": "https://preview.redd.it/whats-the-likelihood-of-the-hawk-tuah-girl-being-on-next-v0-hsm9dfnudved1.jpeg?auto=webp&s=256567b55894a7288be5b7a596e12253a1208c4a",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://www.tampabay.com/resizer/v2/haliey-welch-also-known-as-hawk-tuah-girl-GUE57I7KGTDRYBUIWOX4ZAJXGU.jpg?auth=1445b42ab0feca28b8dacb51e24a18ebd481b4c3032f0f9496a591859c95edbd&height=506&width=900&smart=true",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://www.hollywoodreporter.com/wp-content/uploads/2024/07/0U9A2440.jpg?w=1296",
        "method": "alternative",
        "parameters": (1.05, 3),
        "quantity": 1,
    },
    {
        "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRHhQWN-6jlDMgtCC2BxowaI7U1q15XDxjskg&s",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSW9f0DEiT5nNAuD6Il3_s4YYZREFmQ_9b1tQ&s",
        "method": "alternative",
        "parameters": (1.05, 3),
        "quantity": 1,
    },
    {
        "url": "https://thespun.com/.image/t_share/MjA4MTA1MDE4MTAyOTE2NzUy/screenshot-2024-07-25-at-33956pm.png",
        "method": "alternative",
        "parameters": (1.05, 3),
        "quantity": 1,
    },
    {
        "url": "https://people.com/thmb/FiEWf4Vz9DA-R8V18nXtCzeDexo=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(776x234:778x236)/Haliey-Welch-Hawk-Tuah-girl-091624-tout-ffd7f3e928254867988657a6b18ab642.jpg",
        "method": "alternative",
        "parameters": (1.02, 3),
        "quantity": 1,
    },
    {
        "url": "https://thespun.com/.image/t_share/MjEwNDUzNTQ4OTAxNjA3NDAx/screenshot-2024-11-03-at-85430pm.png",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
]
