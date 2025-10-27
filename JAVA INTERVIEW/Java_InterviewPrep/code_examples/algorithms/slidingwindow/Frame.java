package slidingwindow;

import java.io.Serializable;

public class Frame implements Serializable{
    int seqNo;
    char data;
    Frame(int seqNo, char data){
        this.seqNo = seqNo;
        this.data = data;
    }
}
