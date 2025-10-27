package slidingwindow;
import java.util.*;
import java.io.*;
import java.net.*;

public class Sender{
    Socket socket;
    ObjectOutputStream out;
    ObjectInputStream in;
    int base =0, nextSeqNo=0;
    final int WINDOW_SIZE = 4;
    
    public void run(){
        try{
            socket = new Socket("localhost", 2004);
            out = new ObjectOutputStream(socket.getOutputStream());
            in = new ObjectInputStream(socket.getInputStream());

            Scanner sc = new Scanner(System.in);
            System.out.print("Enter a message to send: ");
            String msg = sc.nextLine();
            int n = msg.length();
            sc.close();

            while(base<n){
                while(nextSeqNo<base+WINDOW_SIZE && nextSeqNo<n){
                    Frame f = new Frame(nextSeqNo, msg.charAt(nextSeqNo));
                    out.writeObject(f);
                    System.out.println("Sent Frame"+f.seqNo+": "+f.data);
                    nextSeqNo++;
                }
                int ack = (Integer) in.readObject();
                System.out.println("Recieved Ack for Frame"+ack);
                base = ack+1;

            }

            out.writeObject(new Frame(-1, '1'));
            System.out.println("All frames sent");
        }catch(Exception e){
            System.out.println("Error occured: "+e.getMessage());

        }finally{
            try{
                socket.close();
            }catch(Exception e){
                System.out.println("Error occured: "+e.getMessage());
            }
        }
    }
    public static void main(String[] args) {
        new Sender().run();
    }
}