t=int(input())
for i in range (t):
    a,b=map(int,input().split())
    if a!=0:
     if (a%2==0 and b==0 ) or (a%2==0 and b%2!=0) or (a%2==0 and b%2==0):
        print("YES")
     else:
        print("NO")
    elif a==0 :
       if b%2==0 or b==0:
          print("YES")
       else:
          print("NO")
    else:
       print("NO")