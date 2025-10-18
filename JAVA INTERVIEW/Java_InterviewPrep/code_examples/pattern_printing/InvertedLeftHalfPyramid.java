// Given an integer N, print N rows of inverted right half pyramid pattern. In inverted right half pattern of N rows, the first row has N number of stars, second row has (N - 1) number of stars and so on till the Nth row which has only 1 star.
import java.util.Scanner;


public class InvertedLeftHalfPyramid {
   public static void main(String[] args) {
       Scanner sc = new Scanner(System.in);
       System.out.print("Enter the number of rows: ");
       int n = sc.nextInt();
       for(int i=0; i<n; i++){
           for(int j=n; j>=i+1; j--){
               System.out.print("* ");
           }
           System.out.println();
       }


   }
}
