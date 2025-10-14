class LRUCache {

    class Node{
        Node prev;
        Node next;
        int key, val;

        Node(int key, int val){
            this.key = key;
            this.val = val;
        }
    }

    HashMap<Integer, Node> map = new HashMap<>();
    Node head = new Node(-1, -1);
    Node tail = new Node(-1, -1);
    
    int cap;

    private void add(Node addNode){
        Node temp = head.next;
        temp.prev = head;

        addNode.next = temp;
        addNode.prev = head;

        temp.prev = addNode;
        head.next = addNode;
    } 

    private void del(Node delNode){
        delNode.prev.next = delNode.next;
        delNode.next.prev = delNode.prev;
    }

    public LRUCache(int capacity) {
        cap = capacity;
        head.next = tail;
        tail.prev = head;
    }
    
    public int get(int key) {
        if(map.containsKey(key)){
            Node ans = map.get(key);
            int ansNo = ans.val;

            map.remove(key);
            del(ans);
            add(ans);

            map.put(key, head.next);

            return ansNo;
        }
        return -1;

    }
    
    public void put(int key, int value) {
        if(map.containsKey(key)){
            Node cur = map.get(key);
            map.remove(key);
            del(cur);
        }

        if(map.size() == cap){
            map.remove(tail.prev.key);
            del(tail.prev);
        }

        add(new Node(key, value));
        map.put(key, head.next);
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */