package cic.cs.unb.ca.jnetpcap.worker;

import org.apache.commons.io.filefilter.FalseFileFilter;
import org.zeromq.SocketType;
import org.zeromq.ZMQ;

import cic.cs.unb.ca.jnetpcap.FlowFeature;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.nio.charset.StandardCharsets;

public class ZmqPubWorker implements Runnable {
    // Hàng đợi chứa các chuỗi flow đã dump thô
    private static final ArrayBlockingQueue<String> queue = new ArrayBlockingQueue<>(1000000);
    private static final int BATCH_SIZE = 256;
    private static final long BATCH_TIMEOUT_MS = 300;

    public static void offer(String flowDump) {
        // Không dùng put() vì nó sẽ làm dừng luồng capture nếu queue đầy.
        // offer() sẽ bỏ qua gói tin nếu hệ thống quá tải (chống treo máy khi DDoS).
        queue.offer(flowDump);
    }

    private volatile boolean running = true;
    private volatile boolean isHeaderServerRunning = true;
    private Thread threadHeader;

    public ZmqPubWorker(){
        // super();
    }

    public void stop(){
        this.stopHeaderServer();
        this.running = false;
    }

    @Override
    public void run() {
        long starttime = System.currentTimeMillis();
        ZMQ.Context context = ZMQ.context(1);
        ZMQ.Socket socket = context.socket(SocketType.PUSH);
        
        socket.setLinger(0); // Đóng socket ngay lập tức khi stop
        socket.setSndHWM(1000); // High Water Mark cho buffer gửi
        socket.connect("tcp://127.0.0.1:5555");        
        
        
        // Khởi động thread nhận req header
        // this.handshake(context);
        this.Init_Listener(context);

        
        // INIT VARIABLES
        int total =0;
        StringBuilder batchBuffer = new StringBuilder(2 * 1024* 1024);
        java.util.List<String> tempBuffer = new java.util.ArrayList<>(BATCH_SIZE);
        int count = 0;
        long lastFlush = System.currentTimeMillis();

        try{
            while (this.running || !queue.isEmpty()) {
                
                    int numDrained = queue.drainTo(tempBuffer, BATCH_SIZE - count);
                    if (numDrained > 0) {
                        for (String flow : tempBuffer) {
                            batchBuffer.append(flow).append('\n');
                        }
                        count += numDrained;
                        tempBuffer.clear();
                    }

                    long now = System.currentTimeMillis();

                    // 3. Kiểm tra gửi ngay sau khi lấy dữ liệu
                    if (count > 0 && (count >= BATCH_SIZE || (now - lastFlush) >= BATCH_TIMEOUT_MS || !this.running)) {
                        socket.send(batchBuffer.toString().getBytes(StandardCharsets.UTF_8), ZMQ.DONTWAIT);
                        
                        // TOTAL COUNT
                        total += count;

                        batchBuffer.setLength(0);
                        // batchBuffer.append(header).append("\n");
                        count = 0;
                        lastFlush = now;
                    }

                    // 4. Nghỉ ngơi nếu thực sự rỗng để tiết kiệm CPU
                    if (numDrained == 0 && count == 0) {
                        Thread.sleep(1); 
                    }
            }
        }
        catch (InterruptedException e) {
            this.running = false;
            Thread.currentThread().interrupt();
        }
        
        finally{
            long deltatime = System.currentTimeMillis() - starttime;
            System.out.println("Flows: " + total + " flows in " + deltatime + " ms. Mean: "+ (float) total / deltatime * 1000.0 + " flows/s");
            System.out.println("[Zmq-Worker] Stopped!");
            socket.close();
            context.term();
        }
    }


    private void Init_Listener(ZMQ.Context context) {
        // Đảm bảo không có luồng cũ đang chạy trước khi khởi tạo
        stopHeaderServer(); 

        ZMQ.Socket headerServer = context.socket(SocketType.REP);
        headerServer.setLinger(0); // Đóng socket ngay lập tức khi close
        headerServer.bind("tcp://127.0.0.1:5556");

        isHeaderServerRunning = true;
        this.threadHeader = new Thread(() -> {
            System.out.println("[Zmq-Worker] Header Server đang chạy tại cổng 5556...");
            try {
                while (isHeaderServerRunning && !Thread.currentThread().isInterrupted()) {
                    // Sử dụng ZMQ.Poller hoặc recv với flag NOBLOCK để không bị kẹt cứng khi tắt
                    byte[] request = headerServer.recv(ZMQ.DONTWAIT); 
                    
                    if (request != null) {
                        String cmd = new String(request, ZMQ.CHARSET);
                        if ("GET_HEADER".equals(cmd)) {
                            String header = FlowFeature.getHeader();
                            headerServer.send(header.getBytes(ZMQ.CHARSET), 0);
                            System.out.println("[Zmq-Worker] Đã gửi Header cho Python.");
                        }
                    } else {
                        // Tránh chiếm dụng CPU 100% khi dùng DONTWAIT
                        Thread.sleep(2500); 
                    }
                }
            } catch (Exception e) {
                System.out.println("[Zmq-Worker] Header Server kết thúc.");
            } finally {
                headerServer.close();
                System.out.println("[Zmq-Worker] Socket 5556 đã đóng.");
            }
        });

        this.threadHeader.setName("ZMQ-Header-Server");
        this.threadHeader.start();
    }
    public void stopHeaderServer() {
        isHeaderServerRunning = false;
        if (this.threadHeader != null && this.threadHeader.isAlive()) 
        {
            try {
                this.threadHeader.interrupt();
                this.threadHeader.join(900); // Đợi tối đa 1s để context end cho an toàn
            }
            catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            
        }
    }
}