bot_dict = {'ありがとう':'ドウイタシマシテ','おはよう':'オハヨウゴザイマス','げんき':'ワタシハトテモゲンキデス'}

while True:
    command = input('bot&gt; ')

    responce = ""

    #辞書のキーが含まれているかチェック
    for key in bot_dict:
        if key in command:
            responce = bot_dict[key]
            break

    # 空文字の判定
    if not responce:
        responce = 'ナニヲイッテイルノデスカ...'

    print(responce)

    if 'ばいばい' in command:
        break
