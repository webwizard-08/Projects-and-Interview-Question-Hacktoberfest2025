/* Pattern to create a  inversed right angled triangle

     *
    **
   ***
  ****
 *****   */

 
#include<stdio.h>

int main(){

int i=0,j=0,k,m=0,n;
printf("Enter No. of rows:");  //Number of rows in right angle triangle.
scanf("%d",&k);
for(n=0;n<k;n++)
{
for(i=0;i<=(k-n-1);i++){
    printf(" ");       //printing leading spaces.
}
for(j=0;j<=n;j++)
{
    printf("*");       //printing the asterisks.
}
    printf("\n");
}
}

