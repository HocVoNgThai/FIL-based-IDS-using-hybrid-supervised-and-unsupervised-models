package cic.cs.unb.ca.ifm;

import cic.cs.unb.ca.flow.FlowMgr;
import cic.cs.unb.ca.jnetpcap.BasicFlow;
import cic.cs.unb.ca.jnetpcap.FlowFeature;
import cic.cs.unb.ca.jnetpcap.worker.InsertCsvRow;
import cic.cs.unb.ca.jnetpcap.worker.ZmqPubWorker;
import cic.cs.unb.ca.jnetpcap.worker.CmdRealTimeFlowWorker;
// import cic.cs.unb.ca.jnetpcap.worker.FlowZmqPublisher;

import org.apache.commons.lang3.StringUtils;
import org.jnetpcap.Pcap;
import org.jnetpcap.PcapIf;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
// json
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
// import java.time.Instant;

//
import java.text.SimpleDateFormat;
import java.time.LocalDate;

//
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.HashMap;
import java.util.Map;


// ===== SUPPORT LIBS ======
import org.json.simple.JSONObject;


public class RT_cmd {

    public static final Logger logger = LoggerFactory.getLogger(RT_cmd.class);

    private static List<PcapIf> pcapIfs = new ArrayList<>();
    private static CmdRealTimeFlowWorker worker;
    private static ZmqPubWorker zworker;
    private static Thread workerThread, zmqThread;

    // private static ExecutorService csvWriterThread;
    // private static ExecutorService logsThread;

    private static String logDir = "";
    // private static BufferedWriter bufferedWriter;
    // private static FlowZmqPublisher publishZmq = new FlowZmqPublisher();
    private static String pcapIfName;

    private static long flowTimeout = 3000L; //36000L
    private static long activityTimeout = 120_000_000L;

    private static volatile boolean running = true;

    public static void main(String[] args) throws Exception {

        if (args.length < 1) {
            helper();
            return;
        }

        parseArgs(args);
        getInputInfo();
        
        // =======================================
        // Trường hợp cần logdir
        // LocalDate currDate = LocalDate.now();
        // logDir += "/"+currDate.getYear()+"-"+currDate.getMonthValue()+"-"+currDate.getDayOfMonth();
        // File directory = new File(logDir);
        // if(directory.mkdirs()){
        //     System.out.println("[INFO] Current Logs Dir: "+ logDir);
        // }
        // else System.out.println("[INFO] Không thể tạo thư mục");
        // bufferedWriter = new BufferedWriter(new FileWriter(logDir+"/cicflowmeter.jsonl"));

        // 
        FlowMgr.getInstance().init();
        // csvWriterThread = Executors.newSingleThreadExecutor();
        // csvWriterThread = new ThreadPoolExecutor(
        //     1, 1, 0L, TimeUnit.MILLISECONDS,
        //     new LinkedBlockingQueue<>(10000),
        //     new ThreadPoolExecutor.DiscardOldestPolicy()
        // );

        // logsThread = new ThreadPoolExecutor(
        //     1, 1, 0L, TimeUnit.MILLISECONDS,
        //     new LinkedBlockingQueue<>(10000),
        //     new ThreadPoolExecutor.DiscardOldestPolicy()
        // );

        startRealtimeFlow();
        registerShutdownHook();

        // block main thread
        while (running) {
            Thread.sleep(100); // 500 (ms)
        }
    }

    /* ======================= CORE ======================= */

    private static void startRealtimeFlow() {

        worker = new CmdRealTimeFlowWorker(
                pcapIfName,
                flowTimeout,
                activityTimeout
        );

        zworker = new ZmqPubWorker();


        workerThread = new Thread(worker, "pcap-worker");
        zmqThread = new Thread(zworker, "zmq-pub");

        workerThread.start();
        zmqThread.start();

        System.out.println("[CICFlowMeter] Rt-Worker started!");
        System.out.println("[CICFlowMeter] Pcap-Worker started!");
    }

    // public static void insertRtFlow(BasicFlow flow) {
    //     final String flowDump = flow.dumpFlowNoConvert();
    //     final String header   = FlowFeature.getHeader();

    //     csvWriterThread.execute(() -> 
    //     {    
    //         // Thread này bây giờ sẽ an toàn trước DDoS
    //         // Nếu queue đầy, task cũ nhất sẽ bị hủy tự động
    //         // System.out.println("[RUN] Flow: "+ flowDump.substring(0,60));
    //         System.out.println("[RUN] Flow: "+ flowDump.length());
    //         sendFlow(header, flowDump);
    //     });

    //     // logsThread.execute(() -> 
    //     // {
    //     //     try{
    //     //         String entry_2 = String.format(
    //     //         "{\"flow_id\":\"%s\",\"event\":\"%s\",\"timestamp\":%d,\"component\":\"cicflowmeter_java\"}%n",
    //     //         flowId, "extract_end",  System.currentTimeMillis()
    //     //         );

    //     //         String entry = String.format(
    //     //         "{\"flow_id\":\"%s\",\"event\":\"%s\",\"timestamp\":%d,\"component\":\"cicflowmeter_java\"}%n",
    //     //         flowId, "start_pcap", flow.getLongTimeStamp()
    //     //         );
                
    //     //         bufferedWriter.write(entry);
    //     //         bufferedWriter.write(entry_2);
    //     //         // Không gọi flush() ở đây để tăng tốc độ ghi file
                
    //     //         // bufferedWriter.flush();
    //     //     }
    //     //     catch (IOException e)
    //     //     {
    //     //         System.out.println("[INFO] ERROR at logsThread" + e);
    //     //     }

    //     // }
    //     // );
    // }

    // private static void sendFlow(String header, String flowDump){
    //     // JSONObject data = new JSONObject();
    //     Map<String, String> data = new HashMap<String, String>();

    //     data.put("header", header);
    //     data.put("flowDump", flowDump);

    //     JSONObject obj = new JSONObject(data);

    //     FlowZmqPublisher.send(obj.toJSONString().getBytes(java.nio.charset.StandardCharsets.UTF_8), 0);
    // }

    private static void stopRealtimeFlow() {
        System.out.println("[INFO] Stopping worker Realtime Packet Capture ...");
        running = false;

        if (worker != null) worker.stop(); // Cái hàm thuộc CmdRealtimeFlowWorker, k có cái này là capture mãi luôn.
        if (zworker !=null) zworker.stop();

        if (workerThread != null) workerThread.interrupt();
        if (zmqThread !=null) zmqThread.interrupt();

        System.out.println("[INFO] Stopping worker, zmqworker...");
    }

    private static void shutdown() {
        // System.out.println("[INFO] Flush and Stop csv/zmqThread. On going ...");

        // csvWriterThread.shutdown();
        // logsThread.shutdown();

        // try {
        //     boolean csvFinished = csvWriterThread.awaitTermination(10, TimeUnit.SECONDS);
        //     // boolean logsFinished = logsThread.awaitTermination(5, TimeUnit.SECONDS);
        //     // Đợi các flow cuối được gửi qua ZMQ
        //     if (!csvFinished) { // || !logsFinished
                
        //         // logger.warn("Forcing CSV/ZMQ writer shutdown");
        //         System.out.println("[WARN] Forcing CSV/ZMQ writer shutdown. Timeout reached, some flows may be lost.");

        //         csvWriterThread.shutdownNow();
        //         // logsThread.shutdownNow();
        //     }

        // } catch (InterruptedException e) {
        //     csvWriterThread.shutdownNow();
        //     // logsThread.shutdownNow();
        //     Thread.currentThread().interrupt();
        // }

        // finally{
        //     System.out.println("[INFO] Successfully Stopped csv/zmqThread!");
            // try {
            //     if (bufferedWriter != null) {
            //         bufferedWriter.flush();
            //         bufferedWriter.close();
            //         System.out.println("[INFO] Closed BufferedWriter.");
            //     }
            // } catch (IOException e) {
            //     System.out.println("[INFO] Error closing bufferedWriter" + e);
            // }

        // }
        // FlowZmqPublisher.close();
    }

    /* ======================= CLI ======================= */

    private static void registerShutdownHook() {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("[INFO] Shutdown signal received");

            try {
                stopRealtimeFlow();
                shutdown();
                System.out.println("[INFO] Hooked! RT_cmd exit cleanly");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }));
    }

    private static void parseArgs(String[] args) throws Exception {

        for (int i = 0; i < args.length; i++) {

            switch (args[i]) {

                case "-h":
                case "--help":
                    helper();
                    System.exit(0);
                    break;

                case "-l":
                case "--list-interface":
                    listPcapIfs();
                    System.exit(0);
                    break;

                case "-i":
                case "--interface":
                    pcapIfName = args[++i];
                    break;

                case "-o":
                case "--out-log-dir":
                    logDir = args[++i];
                    break;

                case "-fto":
                case "--flowTimeout":
                    flowTimeout = Long.parseLong(args[++i]);
                    break;

                case "-ato":
                case "--activityTimeout":
                    activityTimeout = Long.parseLong(args[++i]);
                    break;
            }
        }
    }

    private static void getInputInfo(){
        System.out.println("[INFO] ============= CÁC THÔNG SỐ =============\n- INTERFACE: "+pcapIfName+ "\n- LOG_DIR: "+logDir +"\n- FLOW TIMEOUT: "+flowTimeout+"ms\n- ACT TIMEOUT: "+activityTimeout+"ms");
    }
    private static void listPcapIfs() {

        StringBuilder errbuf = new StringBuilder();
        if (Pcap.findAllDevs(pcapIfs, errbuf) != Pcap.OK) {
            throw new RuntimeException(errbuf.toString());
        }

        System.out.println("[DEBUG ]Available "+ pcapIfs.size() +" interfaces:");
        for (PcapIf p : pcapIfs) {
            System.out.println("[INFO] ============= CÁC Interface network =============\n- " + p.getName() + " : " + p.getDescription());
        }
    }

    private static void helper() {
        System.out.println("CICFlowMeter RT_cmd usage:");
        System.out.println(" -i <interface>");
        System.out.println(" -fto <flow timeout>");
        System.out.println(" -ato <activity timeout>");
        System.out.println(" -o <log dir>");
        System.out.println(" -l --list-interface");
    }
}
