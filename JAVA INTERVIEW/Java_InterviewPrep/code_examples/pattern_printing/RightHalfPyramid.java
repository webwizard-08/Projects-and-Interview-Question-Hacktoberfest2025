// Given an integer N, print N rows of right half pyramid pattern. In right half pattern of N rows, the first row has 1 star, second row has 2 stars and so on till the Nth row which has N stars. All the stars are left aligned.
import java.util.Scanner;
public class RightHalfPyramid {
   public static void main(String[] args) {
       Scanner sc = new Scanner(System.in);
       System.out.print("Enter the number of rows: ");
       int n = sc.nextInt();
       for(int i=0; i<n; i++){
           for(int j=0; j<i+1; j++){
               System.out.print("* ");
           }
           System.out.println();
       }


   }
}
