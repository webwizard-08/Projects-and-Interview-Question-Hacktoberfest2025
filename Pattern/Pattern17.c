/* Pattern to create a diamond

      *
     ***
    *****
   *******
  *********
 ***********
*************
 ***********
  *********
   *******
    *****
     ***
      *       */
     

#include<stdio.h>

int main(){

int i=0,j=0,k,m=0,n,p;
printf("Enter No. of rows:");   //Number of rows in a triangle.
scanf("%d",&k); 
for(n=0;n<k;n++)
{
    //Loop for printing upper triangle.
    for(i=0;i<=(k-n-1);i++){
    printf(" ");    //printing leading spaces.
}
for(j=0;j<=(2*n);j++)
{
    printf("*");       //printing the asterisks.
}
    printf("\n");
}
//Loop for printing lower triangle.
for(p=k;p>=0;p--)
{
    for(i=0;i<=(k-p-1);i++){
    printf(" ");      //printing leading spaces.
}
for(j=0;j<=(2*p);j++)
{
    printf("*");       //printing the asterisks.
}
    printf("\n");
}
}
