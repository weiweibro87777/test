from random import randint

lowest = 1
highest = 100
answer = randint(lowest, highest)

while True:
    guess = input('密碼介於 ' + str(lowest) + '-' + str(highest) + ':\n>>')
    
    try:
        guess = int(guess)
    except ValueError:
        print('格式錯誤，請輸入數字\n')
        continue
    
    if guess <= lowest or guess >= highest:
        print('請輸入 ' + str(lowest) + '-' + str(highest) + ' 之間的整數\n')
        continue
    
    if guess == answer:
        print('答對了！')
        break
    elif guess < answer:
        lowest = guess
    else:
        highest = guess