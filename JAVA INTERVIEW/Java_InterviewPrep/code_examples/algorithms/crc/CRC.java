package crc;
import java.util.*;
public class CRC{
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the no of databits: ");
        int n = sc.nextInt();
        System.out.print("Enter the no of generator bits: ");
        int m = sc.nextInt();

        int[] d=new int[n+m], g=new int[m];
        System.out.print("Enter the data bits: ");
        for(int i=0; i<n; i++) d[i] = sc.nextInt();
        Arrays.fill(d, n, n+m, 0);

        System.out.print("Enter the generator bits: ");
        for(int i=0; i<m; i++) g[i] = sc.nextInt();

        int[] r=new int[n+m], z=new int[m];

        for(int i=0; i<m; i++){
            r[i]=d[i];
            z[i]=0;
        }

        for(int i=0; i<n; i++){
            int k=0; int msb = r[i];
            for(int j=i; j<m+i; j++){
                if(msb==0) r[j] = r[j]^z[k];
                else r[j] = r[j]^g[k];
            }
            r[m+i] = d[m+i];
        }

        System.out.print("The added code bits are: ");
        for(int i=n; i<n+m-1; i++){
            d[i] = r[i];
            System.out.print(r[i] + "\t");
        }
        System.out.println("\nThe code bits are:");
        for(int i=0; i<n+m-1; i++){
            System.out.print(d[i] + "\t");
        }

    }
}