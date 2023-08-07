   
def repeat():
    cache= []
    requests= []


    def fifo():
        for p in requests:
            if p not in cache: 
                print('miss')
                cache.append(p)
            else: 
                print('hit')
            while len(cache) > 8:
                cache.pop(0)
        #print(cache)

    def lfu():
        frequency_dictionary= {}
        for requestedPage in requests:
            if requestedPage in cache:
                print('hit')
            else:
                if len(cache) <8:
                    print('miss')
                    cache.append(requestedPage)
                else:
                    print('miss')
                    infiniteValue = float('inf')
                    leastRequestedPage = float('inf')
                    for eachElement in cache:
                        if(frequency_dictionary[eachElement] < infiniteValue):
                            leastRequestedPage = eachElement
                            infiniteValue = frequency_dictionary[eachElement]
                        elif (frequency_dictionary[eachElement] == infiniteValue):
                            if (eachElement < leastRequestedPage):
                                leastRequestedPage = eachElement
                    cache.remove(leastRequestedPage)
                    cache.append(requestedPage)
            if requestedPage not in frequency_dictionary.keys():
                frequency_dictionary[requestedPage] = 1
            else:
                frequency_dictionary[requestedPage] += 1


    while True:
            b= int(input('Please requests a page from memory.'))
            if b != 0:
                requests.append(b)
            else:
                break
            #if b =='0':
            #    print(requests)

    choice = input("Enter 1 for FIFO\nEnter 2 for LFU\nEnter Q for quit program\n")
        

    if choice == '1':
                print(f'fifo')
                fifo()
                print(cache)
                cache.clear()
                requests.clear()
                repeat()
    elif choice == '2':
                print(f'lfu')
                lfu()
                print(cache)
                cache.clear()
                requests.clear()
                repeat()
    elif choice == 'Q':
                exit()   
                


repeat()