package cic.cs.unb.ca.ifm;

import cic.cs.unb.ca.flow.FlowMgr;
import cic.cs.unb.ca.jnetpcap.BasicFlow;
import cic.cs.unb.ca.jnetpcap.FlowFeature;
import cic.cs.unb.ca.jnetpcap.worker.InsertCsvRow;
import cic.cs.unb.ca.jnetpcap.worker.CmdRealTimeFlowWorker;
import cic.cs.unb.ca.jnetpcap.worker.FlowZmqPublisher;

import org.apache.commons.lang3.StringUtils;
import org.jnetpcap.Pcap;
import org.jnetpcap.PcapIf;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.HashMap;
import java.util.Map;


// ===== SUPPORT LIBS ======
import org.json.simple.JSONObject;


public class RT_cmd {

    public static final Logger logger = LoggerFactory.getLogger(RT_cmd.class);

    private static List<PcapIf> pcapIfs = new ArrayList<>();
    private static CmdRealTimeFlowWorker worker;
    private static Thread workerThread;

    private static ExecutorService csvWriterThread;
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

        FlowMgr.getInstance().init();
        csvWriterThread = Executors.newSingleThreadExecutor();

        startRealtimeFlow();
        startKeyListener();

        // block main thread
        while (running) {
            Thread.sleep(50); // 500 (ms)
        }
    }

    /* ======================= CORE ======================= */

    private static void startRealtimeFlow() {

        worker = new CmdRealTimeFlowWorker(
                pcapIfName,
                flowTimeout,
                activityTimeout
        );

        workerThread = new Thread(worker, "pcap-worker");
        workerThread.start();

        logger.info("Realtime traffic flow started");
    }

    public static void insertRtFlow(BasicFlow flow) {
        String flowDump = flow.dumpFlowBasedFeaturesEx();
        String header  = FlowFeature.getHeader();

        csvWriterThread.execute(() -> 
        {
            System.out.println("[RUN] Flow: "+ flowDump.substring(0,60));
            sendFlow(header, flowDump);
        });
    }

    private static void sendFlow(String header, String flowDump){
        // JSONObject data = new JSONObject();
        Map<String, String> data = new HashMap<String, String>();

        data.put("header", header);
        data.put("flowDump", flowDump);

        JSONObject obj = new JSONObject(data);

        FlowZmqPublisher.send(obj.toJSONString().getBytes(java.nio.charset.StandardCharsets.UTF_8), 0);
    }

    private static void stopRealtimeFlow() {
        System.out.println("[INFO] Stopping worker Realtime Packet Capture ...");

        running = false;

        if (worker != null) {
            // Cái hàm thuộc CmdRealtimeFlowWorker, k có cái này là capture mãi luôn.
            worker.stop();
        }

        if (workerThread != null) {
            // exit thread
            workerThread.interrupt();
        }

        System.out.println("[INFO] Successfully Stopped worker!");
    }

    private static void shutdown() {

        // logger.info("Shutting down CSV writer...");
        System.out.println("[INFO] Flush and Stop csv/zmqThread. On going ...");

        csvWriterThread.shutdown();

        try {
            // 4️⃣ Đợi các flow cuối được gửi qua ZMQ
            if (!csvWriterThread.awaitTermination(10, TimeUnit.SECONDS)) {
                
                logger.warn("Forcing CSV/ZMQ writer shutdown");
                System.out.println("[WARN] Forcing CSV/ZMQ writer shutdown");

                csvWriterThread.shutdownNow();
            }
        } catch (InterruptedException e) {
            csvWriterThread.shutdownNow();
            Thread.currentThread().interrupt();
        }

        System.out.println("[INFO] Successfully Stopped csv/zmqThread!");
        FlowZmqPublisher.close();
            
    }

    /* ======================= CLI ======================= */

    private static void startKeyListener() {

        Thread keyThread = new Thread(() -> {
            try {
                System.out.println("[INFO] Press 'q' then ENTER to stop...");
                while (true) {
                    int c = System.in.read();
                    if (c == 'q' || c == 'Q') {
                        System.out.println("[INFO] Stop signal received");
                        stopRealtimeFlow();
                        shutdown();
                        System.out.println("[INFO] RT_cmd exited cleanly");
                        break;
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

        keyThread.setDaemon(true);
        keyThread.start();
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
        System.out.println("[INFO] ============= CÁC THÔNG SỐ =============\n- INTERFACE: "+pcapIfName+"\n- FLOW TIMEOUT: "+flowTimeout+"ms\n- ACT TIMEOUT: "+activityTimeout+"ms");
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
        System.out.println(" -o <output dir>");
        System.out.println(" -l --list-interface");
    }
}
