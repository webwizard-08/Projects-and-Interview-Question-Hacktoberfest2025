t=int(input())
for i in range(t):
    n=int(input())
    a=n//3
    b=n%3
    if b!=0:
        if b>=2:
            print(a+(b%2),a+(b//2))
        else:
            print(a+1,a)
    else:
        print(a,a)
