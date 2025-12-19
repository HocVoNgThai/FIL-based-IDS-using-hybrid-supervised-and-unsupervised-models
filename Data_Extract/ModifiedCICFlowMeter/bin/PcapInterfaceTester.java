import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

import javax.swing.SwingWorker;

import org.jnetpcap.Pcap;
import org.jnetpcap.PcapIf;
// import org.slf4j.Logger;
// import org.slf4j.LoggerFactory;

public class PcapInterfaceTester {

    public static void main(String[] args) {
        System.out.println("--- Bắt đầu Pcap Interface Loader ---");
        // ⭐ THÊM KIỂM TRA ĐƯỜNG DẪN THƯ VIỆN NATIVE
        String libPath = System.getProperty("java.library.path");
        System.out.println("[DEBUG] java.library.path đang sử dụng: " + libPath);
        
        // Tạo và thực thi SwingWorker
        LoadPcapInterfaceWorker worker = new LoadPcapInterfaceWorker();
        worker.execute();

        try {
            // Lấy kết quả một cách đồng bộ (chặn luồng chính)
            List<PcapIf> interfaces = worker.get(); 
            
            System.out.println("\n=== Giao diện mạng được tìm thấy ===");
            if (interfaces.isEmpty()) {
                System.out.println("Không tìm thấy giao diện mạng nào.");
            } else {
                System.out.printf("Tổng cộng tìm thấy: %d\n", interfaces.size());
                for (int i = 0; i < interfaces.size(); i++) {
                    PcapIf iface = interfaces.get(i);
                    System.out.printf("[%d] Tên: %s | Mô tả: %s\n", 
                        i + 1, 
                        iface.getName(), 
                        iface.getDescription() != null ? iface.getDescription() : "N/A"
                    );
                }
            }

        } catch (InterruptedException e) {
            System.err.println("Worker bị gián đoạn: " + e.getMessage());
        } catch (ExecutionException e) {
            // Bắt lỗi được ném ra từ doInBackground
            System.err.println("\n=== LỖI trong quá trình tải giao diện Pcap ===");
            System.err.println("Nguyên nhân gốc: " + e.getCause().getMessage());
            System.err.println("Đảm bảo thư viện native JNetPcap có thể truy cập (kiểm tra java.library.path).");
        }
        
        System.out.println("\n--- Tester hoàn tất ---");
    }
}

/**
 * Class Worker thực hiện việc quét giao diện mạng ở luồng nền.
 * (Mã dựa trên đoạn bạn cung cấp)
 */
class LoadPcapInterfaceWorker extends SwingWorker<List<PcapIf>,String>{
    
    public LoadPcapInterfaceWorker() {
        super();
    }

    @Override
    protected List<PcapIf> doInBackground() throws Exception {
        
        StringBuilder errbuf = new StringBuilder();
        List<PcapIf> ifs = new ArrayList<>();
        
        // Gọi API của JNetPcap để tìm kiếm tất cả các thiết bị mạng
        if(Pcap.findAllDevs(ifs, errbuf)!=Pcap.OK) {
            throw new Exception(errbuf.toString());
        }
        return ifs;
    }

    @Override
    protected void done() {
        // Logic khi tác vụ hoàn thành (thường là cập nhật GUI)
        super.done();
    }
}