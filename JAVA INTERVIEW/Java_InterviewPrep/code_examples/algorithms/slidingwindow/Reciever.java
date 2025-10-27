package slidingwindow;
import java.io.*;
import java.net.*;
import java.util.*;
public class Reciever {
    Socket socket;
    ServerSocket server;
    ObjectOutputStream out;
    ObjectInputStream in;

    public void run(){
        try{
            server = new ServerSocket(2004);
            socket = server.accept();
            out =new  ObjectOutputStream(socket.getOutputStream());
            in =new ObjectInputStream(socket.getInputStream());
            int expectedFrame =0;

            while(true){
                Frame f =(Frame) in.readObject();
                if(f.seqNo == -1){
                    System.out.println("Terminating");
                    break;
                }else if(f.seqNo == expectedFrame){
                    System.out.println("Received Frame"+f.seqNo+": "+f.data);
                    out.writeObject(f.seqNo);
                    expectedFrame++;
                }else{
                    System.out.println("Recieved corrupted frame..");
                    out.writeObject(expectedFrame -1);

                }
            }

        }catch(Exception e){
            System.out.println("Error: "+e.getMessage());
        }finally{
            try{
                socket.close();
                server.close();

            }catch(Exception e){
            System.out.println("Error: "+e.getMessage());
            }
        }

    }
    public static void main(String[] args) {
        new Reciever().run();
    }
}