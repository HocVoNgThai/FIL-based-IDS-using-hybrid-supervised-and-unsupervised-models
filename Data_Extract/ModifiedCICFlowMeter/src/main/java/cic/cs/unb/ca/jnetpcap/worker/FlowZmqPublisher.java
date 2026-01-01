package cic.cs.unb.ca.jnetpcap.worker;

import org.zeromq.SocketType;
import org.zeromq.ZMQ;

public class FlowZmqPublisher {

    // private static final ZMQ.Context context = ZMQ.context(1);
    // private static final ZMQ.Socket socket = context.socket(ZMQ.PUSH);
    // private static final ZMQ.Socket socket = context.socket(SocketType.PUSH);
    
    // static {
    //     socket.setLinger(1000); // 1s flush rồi drop
    //     // socket.setSendTimeOut(1000);
    //     socket.setSndHWM(1000000);
    //     socket.setSendBufferSize(2 * 1024 * 1024);  // 4MB TCP buffer

    //     socket.bind("tcp://*:5555");
    //     // System.out.println("[ZMQ] Connected to Python");
    // }

    // public static void send(byte[] msg , int flags) {
    //     socket.send(msg, ZMQ.DONTWAIT);
    // }

    // public static void sendBatch(byte[][] messages) {
    //     if (messages.length == 0) return;
    //     for (int i = 0; i < messages.length - 1; i++) {
    //         socket.sendMore(messages[i]);
    //     }
    //     socket.send(messages[messages.length - 1]);
    // }


    // public static void close() {
    //     socket.close();
    //     context.term();
    //     return;
    // }
}