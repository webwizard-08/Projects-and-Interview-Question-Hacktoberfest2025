t=int(input())
for i in range(t):
  x,y,n=map(int,input().split())
  if x>0:
    a=n%x
 
    if a==y:
        print(n)
    else:
        if a<y:
            print(n-(x+a-y))
        else:
            print(n-a+y)