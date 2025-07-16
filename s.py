def hashc(a):
    l=[]
    for i in a:
        if i=="#":
            continue
        l.append(i)
    return ''.join(l)
a='Move#Hash#to#Front'
print(hashc(a))
            