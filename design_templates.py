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

# urls = [
#     {
#         "url": "https://d.newsweek.com/en/full/2463914/haliey-welch.jpg?w=1200&f=daddd6871e66c2facf472a455c0a94d6",
#         "method": "default",
#         "parameters": (1.3, 5),
#         "quantity": 1,
#     },
#     {
#         "url": "https://www.rollingstone.com/wp-content/uploads/2024/12/hailey-welch-hawk-memecoin.jpg?w=1581&h=1054&crop=1",
#         "method": "default",
#         "parameters": (1.3, 5),
#         "quantity": 1,
#     },
#     {
#         "url": "https://content.api.news/v3/images/bin/7a30da19649e6f81aac060bdaa9f59c3",
#         "method": "alternative",
#         "parameters": (1.05, 3),
#         "quantity": 1,
#     },
#     {
#         "url": "https://www.pajiba.com/assets_c/2024/12/hawktuahscam-thumb-700xauto-267027.png",
#         "method": "default",
#         "parameters": (1.3, 5),
#         "quantity": 1,
#     },
#     {
#         "url": "https://i.ytimg.com/vi/JKGT6T4R3Pg/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLBJZxg3SVov1N9K4Dm0H2AHpCehHw",
#         "method": "default",
#         "parameters": (1.3, 5),
#         "quantity": 1,
#     },
#     {
#         "url": "https://townsquare.media/site/204/files/2024/07/attachment-HawkTuah.jpg",
#         "method": "alternative",
#         "parameters": (1.1, 3),
#         "quantity": 2,
#     },
#     {
#         "url": "https://preview.redd.it/whats-the-likelihood-of-the-hawk-tuah-girl-being-on-next-v0-hsm9dfnudved1.jpeg?auto=webp&s=256567b55894a7288be5b7a596e12253a1208c4a",
#         "method": "default",
#         "parameters": (1.3, 5),
#         "quantity": 1,
#     },
#     {
#         "url": "https://www.tampabay.com/resizer/v2/haliey-welch-also-known-as-hawk-tuah-girl-GUE57I7KGTDRYBUIWOX4ZAJXGU.jpg?auth=1445b42ab0feca28b8dacb51e24a18ebd481b4c3032f0f9496a591859c95edbd&height=506&width=900&smart=true",
#         "method": "default",
#         "parameters": (1.3, 5),
#         "quantity": 1,
#     },
#     {
#         "url": "https://www.hollywoodreporter.com/wp-content/uploads/2024/07/0U9A2440.jpg?w=1296",
#         "method": "alternative",
#         "parameters": (1.05, 3),
#         "quantity": 1,
#     },
#     {
#         "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRHhQWN-6jlDMgtCC2BxowaI7U1q15XDxjskg&s",
#         "method": "default",
#         "parameters": (1.3, 5),
#         "quantity": 1,
#     },
#     {
#         "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSW9f0DEiT5nNAuD6Il3_s4YYZREFmQ_9b1tQ&s",
#         "method": "alternative",
#         "parameters": (1.05, 3),
#         "quantity": 1,
#     },
#     {
#         "url": "https://thespun.com/.image/t_share/MjA4MTA1MDE4MTAyOTE2NzUy/screenshot-2024-07-25-at-33956pm.png",
#         "method": "alternative",
#         "parameters": (1.05, 3),
#         "quantity": 1,
#     },
#     {
#         "url": "https://people.com/thmb/FiEWf4Vz9DA-R8V18nXtCzeDexo=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(776x234:778x236)/Haliey-Welch-Hawk-Tuah-girl-091624-tout-ffd7f3e928254867988657a6b18ab642.jpg",
#         "method": "alternative",
#         "parameters": (1.02, 3),
#         "quantity": 1,
#     },
#     {
#         "url": "https://thespun.com/.image/t_share/MjEwNDUzNTQ4OTAxNjA3NDAx/screenshot-2024-11-03-at-85430pm.png",
#         "method": "default",
#         "parameters": (1.3, 5),
#         "quantity": 1,
#     },
# ]

urls = [
    {
        "url": "https://media.cnn.com/api/v1/images/stellar/prod/121101103247-barack-obama-hedshot.jpg?q=w_1916,h_2608,x_0,y_0,c_fill",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://www.whitehouse.gov/wp-content/uploads/2021/04/V20210305LJ-0043-cropped.jpg?w=1536",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://www.womenshistory.org/sites/default/files/images/2018-07/Clinton_Hillary%20square.jpg",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Joe_Biden_presidential_portrait.jpg/640px-Joe_Biden_presidential_portrait.jpg",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://hips.hearstapps.com/hmg-prod/images/gettyimages-605917960-copy.jpg",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/5a/JimmyCarterPortrait2.jpg",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a5/Official_photo_of_Speaker_Nancy_Pelosi_in_2019.jpg",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://i.abcnewsfe.com/a/509f7246-1774-47d4-a184-9f54e7113fe5/bernie-ap-jt-241106_1730930531849_hpMain.jpg",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Al_Gore%2C_Vice_President_of_the_United_States%2C_official_portrait_1994.jpg/800px-Al_Gore%2C_Vice_President_of_the_United_States%2C_official_portrait_1994.jpg",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://cdn.britannica.com/51/181151-050-AB5C5DB5/Elizabeth-Warren.jpg?w=400&h=300&c=crop",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://cdn.britannica.com/45/192845-050-479AC35E/Andrew-Cuomo-2010.jpg",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://media.vanityfair.com/photos/67508917a7f7305fb65123a7/master/w_2560%2Cc_limit/Alexandria-Ocasio-Cortez-2028-election.jpg",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://cdn.britannica.com/18/132818-050-55173F3F/John-Kerry-2005.jpg",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://hips.hearstapps.com/hmg-prod/images/michael-bloomberg--gettyimages-1340957840.jpg",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
    {
        "url": "https://media-cldnry.s-nbcnews.com/image/upload/t_fit-1240w,f_auto,q_auto:best/rockcms/2024-07/240717-gavin-newsom-al-0937-d93ee7.jpg",
        "method": "default",
        "parameters": (1.3, 5),
        "quantity": 1,
    },
]
