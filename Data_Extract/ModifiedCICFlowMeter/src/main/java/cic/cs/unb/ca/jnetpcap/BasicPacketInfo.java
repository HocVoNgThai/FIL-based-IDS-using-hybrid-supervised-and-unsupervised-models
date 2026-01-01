package cic.cs.unb.ca.jnetpcap;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

import org.jnetpcap.packet.format.FormatUtils;

public class BasicPacketInfo {
	
/*  Basic Info to generate flows from packets  	*/
    private    long id;
    private    byte[] src;
    private    byte[] dst;
    private    int srcPort;
    private    int dstPort;
    private    int protocol;
    private    long   timeStamp;
    private    long   payloadBytes;
    private    String  flowId = null;  
	// new
	private    int intsrcIp;
	private    int intdstIp;
	private	   String fwdFID = null;
	private    String bwdFID = null;
	// private    String preFlowId;
	private    boolean forward = true;
/* ******************************************** */    
    private    boolean flagFIN = false;
	private    boolean flagPSH = false;
	private    boolean flagURG = false;
	private    boolean flagECE = false;
	private    boolean flagSYN = false;
	private    boolean flagACK = false;
	private    boolean flagCWR = false;
	private    boolean flagRST = false;
	private	   int TCPWindow=0;
	private	   long headerBytes;
	private int payloadPacket=0;

	public BasicPacketInfo(byte[] src, byte[] dst, int srcPort, int dstPort,
			int protocol, long timeStamp, IdGenerator generator) {
		super();
		this.id = generator.nextId();
		this.src = src;
		this.dst = dst;
		this.srcPort = srcPort;
		this.dstPort = dstPort;
		this.protocol = protocol;
		this.timeStamp = timeStamp;
		this.intsrcIp =IpToIpv4Int(this.src);
		this.intdstIp = IpToIpv4Int(this.dst);

		generateFlowId();
	}
	
    public BasicPacketInfo(IdGenerator generator) {
		super();
		this.id = generator.nextId();
	}
    
    
	private int IpToIpv4Int(byte[] ip ){
		return ((ip[0] & 0xFF) << 24) |
               ((ip[1] & 0xFF) << 16) |
               ((ip[2] & 0xFF) << 8)  |
               ((ip[3] & 0xFF));
	}

	private String hashFlowId(int srcIp, int dstIp, int srcPort, int dstPort, int protocol) { //,int index

        // Bước 1: Gom dữ liệu vào 2 biến long (không làm mất bit nào)

        long part1 = ((long) srcIp << 32) | (dstIp & 0xFFFFFFFFL);

        long part2 = ((long) srcPort << 48) | ((long) dstPort << 32) | ((long) protocol << 16); // | (index & 0xFFFFL);

       

        // Bước 2: Trộn bit (Mixer) - Sử dụng thuật toán MurmurHash3 đơn giản hóa

        long h = part1 ^ part2 ^ timeStamp;

        h ^= h >>> 33;

        h *= 0xff51afd7ed558ccdL; // Hằng số giúp khuếch tán bit

        h ^= h >>> 33;

        h *= 0xc4ceb9fe1a85ec53L;

        h ^= h >>> 33;

        return Long.toHexString(h); // Trả về lon
	}

	public String generateFlowId(){    
		// if (Integer.compareUnsigned(this.intsrcIp, this.intdstIp) > 0) {
    	// 	forward = false;
		// }	
		this.fwdFID  = this.hashFlowId(this.intsrcIp, this.intdstIp, this.srcPort, this.dstPort, this.protocol);
		this.bwdFID  = this.hashFlowId(this.intdstIp, this.intsrcIp, this.dstPort, this.srcPort, this.protocol);
		
		// for(int i=0; i<this.src.length;i++){           
    	// 	if(((Byte)(this.src[i])).intValue() != ((Byte)(this.dst[i])).intValue()){
    	// 		if(((Byte)(this.src[i])).intValue() >((Byte)(this.dst[i])).intValue()){
    	// 			this.forward = false;
    	// 		}
    	// 		i=this.src.length;
    	// 	}
    	// }  
		
		if (Integer.compareUnsigned(this.intsrcIp, this.intdstIp) > 0) {
    		this.forward = false;
		}

		this.flowId = this.bwdFID;
        
		if(this.forward){
			this.flowId =  this.fwdFID;
			// this.flowId = this.getSourceIP() + "-" + this.getDestinationIP() + "-" + this.srcPort  + "-" + this.dstPort  + "-" + this.protocol;
            return this.flowId;
        }

        return this.flowId;
	}

 	public String fwdFlowId() {  
		// this.flowId = this.getSourceIP() + "-" + this.getDestinationIP() + "-" + this.srcPort  + "-" + this.dstPort  + "-" + this.protocol;
		return this.fwdFID;
	}
	
	public String bwdFlowId() {
		// this.flowId = this.getDestinationIP() + "-" + this.getSourceIP() + "-" + this.dstPort  + "-" + this.srcPort  + "-" + this.protocol;
		return this.bwdFID;
	}
    
	public String dumpInfo() {
		return null;
	}
	public int getPayloadPacket() {
		return payloadPacket+=1;
	}
    
    public String getSourceIP(){
    	return FormatUtils.ip(this.src);
    }

    public String getDestinationIP(){
    	return FormatUtils.ip(this.dst);
    }
    
    
	public long getId() {
		return id;
	}

	public void setId(long id) {
		this.id = id;
	}

	public byte[] getSrc() {
		return Arrays.copyOf(src,src.length);
	}

	public void setSrc(byte[] src) {
		this.src = src;
	}

	public byte[] getDst() {
		return Arrays.copyOf(dst,dst.length);
	}

	public void setDst(byte[] dst) {
		this.dst = dst;
	}

	public int getSrcPort() {
		return srcPort;
	}

	public void setSrcPort(int srcPort) {
		this.srcPort = srcPort;
	}

	public int getDstPort() {
		return dstPort;
	}

	public void setDstPort(int dstPort) {
		this.dstPort = dstPort;
	}

	public int getProtocol() {
		return protocol;
	}

	public void setProtocol(int protocol) {
		this.protocol = protocol;
	}

	public long getTimeStamp() {
		return timeStamp;
	}

	public void setTimeStamp(long timeStamp) {
		this.timeStamp = timeStamp;
	}

	public String getFlowId() {
		return this.flowId!=null?this.flowId:generateFlowId();
	}

	public void setFlowId(String flowId) {		
		this.flowId = flowId;
	}

	public boolean isForwardPacket(byte[] sourceIP) {
		return Arrays.equals(sourceIP, this.src);
	}

	public long getPayloadBytes() {
		return payloadBytes;
	}

	public void setPayloadBytes(long payloadBytes) {
		this.payloadBytes = payloadBytes;
	}

	public long getHeaderBytes() {
		return headerBytes;
	}

	public void setHeaderBytes(long headerBytes) {
		this.headerBytes = headerBytes;
	}

	public boolean hasFlagFIN() {
		return flagFIN;
	}

	public void setFlagFIN(boolean flagFIN) {
		this.flagFIN = flagFIN;
	}

	public boolean hasFlagPSH() {
		return flagPSH;
	}

	public void setFlagPSH(boolean flagPSH) {
		this.flagPSH = flagPSH;
	}

	public boolean hasFlagURG() {
		return flagURG;
	}

	public void setFlagURG(boolean flagURG) {
		this.flagURG = flagURG;
	}

	public boolean hasFlagECE() {
		return flagECE;
	}

	public void setFlagECE(boolean flagECE) {
		this.flagECE = flagECE;
	}

	public boolean hasFlagSYN() {
		return flagSYN;
	}

	public void setFlagSYN(boolean flagSYN) {
		this.flagSYN = flagSYN;
	}

	public boolean hasFlagACK() {
		return flagACK;
	}

	public void setFlagACK(boolean flagACK) {
		this.flagACK = flagACK;
	}

	public boolean hasFlagCWR() {
		return flagCWR;
	}

	public void setFlagCWR(boolean flagCWR) {
		this.flagCWR = flagCWR;
	}

	public boolean hasFlagRST() {
		return flagRST;
	}

	public void setFlagRST(boolean flagRST) {
		this.flagRST = flagRST;
	}

	public int getTCPWindow(){
		return TCPWindow;
	}

	public void setTCPWindow(int TCPWindow){
		this.TCPWindow = TCPWindow;
	}
}
