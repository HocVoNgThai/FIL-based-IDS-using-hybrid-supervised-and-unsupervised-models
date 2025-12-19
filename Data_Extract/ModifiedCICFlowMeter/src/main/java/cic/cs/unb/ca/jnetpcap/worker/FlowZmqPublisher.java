package cic.cs.unb.ca.jnetpcap.worker;

import org.zeromq.SocketType;
import org.zeromq.ZMQ;

public class FlowZmqPublisher {

    private static final ZMQ.Context context = ZMQ.context(1);
    // private static final ZMQ.Socket socket = context.socket(ZMQ.PUSH);
    private static final ZMQ.Socket socket = context.socket(SocketType.PUSH);
    static {
        socket.setLinger(1000); // 1s flush rồi drop
        // socket.setSendTimeOut(1000);

        socket.connect("tcp://127.0.0.1:5555");
        // System.out.println("[ZMQ] Connected to Python");
    }

    public static void send(byte[] msg , int flags) {
        socket.send(msg, flags);
    }

    public static void close() {
        socket.close();
        context.term();
        return;
    }
}