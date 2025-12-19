package cic.cs.unb.ca.jnetpcap.worker;

import cic.cs.unb.ca.jnetpcap.BasicFlow;
import cic.cs.unb.ca.jnetpcap.FlowGenerator;
import cic.cs.unb.ca.jnetpcap.PacketReader;
import org.jnetpcap.Pcap;
import org.jnetpcap.nio.JMemory.Type;
import org.jnetpcap.packet.PcapPacket;
import org.jnetpcap.packet.PcapPacketHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicBoolean;


// ==== Support
import cic.cs.unb.ca.ifm.RT_cmd;


// ===== CLASS =====
public class CmdRealTimeFlowWorker implements Runnable, FlowGenListener {

    public static final Logger logger = LoggerFactory.getLogger(CmdRealTimeFlowWorker.class);

    private String device;
    private long flowTout, actTout;
    private final AtomicBoolean running = new AtomicBoolean(true);
    private Pcap pcap;

    public CmdRealTimeFlowWorker(String device) {
        this.device = device;
    }

    public CmdRealTimeFlowWorker(String device, long flowTout, long actTout) {
        this.device = device;
        this.flowTout = flowTout;
        this.actTout = actTout;
    }

    public void stop() {
        running.set(false);
        if (pcap != null) {
            pcap.breakloop();
        }
    }

    @Override
    public void run() {

        FlowGenerator flowGen = new FlowGenerator(true, flowTout, actTout);
        flowGen.addFlowListener(this);

        StringBuilder errbuf = new StringBuilder();
        pcap = Pcap.openLive(
                device,
                64 * 1024,
                Pcap.MODE_PROMISCUOUS,
                60_000,
                errbuf
        );

        if (pcap == null) {
            logger.error("open {} fail -> {}", device, errbuf);
            return;
        }

        PcapPacketHandler<Void> handler = (packet, user) -> {

            if (!running.get()) {
                pcap.breakloop();
                return;
            }

            PcapPacket permanent = new PcapPacket(Type.POINTER);
            packet.transferStateAndDataTo(permanent);

            flowGen.addPacket(
                    PacketReader.getBasicPacketInfo(permanent, true, false)
            );
        };

        logger.info("Pcap listening on {}", device);
        pcap.loop(Pcap.LOOP_INFINITE, handler, null);

        pcap.close();
        logger.info("Pcap closed");
    }

    @Override
    public void onFlowGenerated(BasicFlow flow) {
        // logger.info("FLOW DONE: {}", flow.getFlowId());
        RT_cmd.insertRtFlow(flow); // hoặc FlowMgr / CsvWriter
    }
}
