# TEMP
🟢 CPU sử dụng (thực tế đo)

Một Python daemon kiểu:

while True:
    time.sleep(1)


→ CPU usage: 0.0%
(top hiển thị 0.0 hoặc 0.1%)

Kể cả worker queue block:

job = queue.get()  # chờ tới khi có job


→ CPU bằng 0 (thread blocking).

🟢 Khi nào Python tốn CPU?

Chỉ khi đang chạy job học incremental:

AutoEncoder training vài epoch

OCSVM partial_fit

XGBoost incremental training

Nhưng vì bạn chạy 1 job/ngày, và job chạy tuần tự (không song song), thì thời gian tiêu CPU rất ngắn.

🟢 Độ ổn định của Python daemon?

✔ Không rò rỉ bộ nhớ nếu bạn không tạo list khổng lồ mỗi vòng lặp
✔ Không bị treo CPU do sleep hoặc queue block
✔ Không gây nghẽn network
✔ Không tạo thread vô hạn

Bạn chạy bằng systemd nên:

crash → auto restart

memory quá lớn → systemd cắt

log theo dõi dễ

chạy nền 24/7 không cần screen/tmux

🟢 Nếu so sánh với Golang, Rust?
Ngôn ngữ	Idle RAM	Idle CPU	Độ ổn định dài hạn
Golang	~5 MB	0%	Rất cao
Rust	~1–3 MB	0%	Rất cao
Python	~60 MB	0%	Cao


# Illustration
        +------------------------+
        |   Incremental Daemon  |
        +------------------------+
            | enqueue mỗi 24h (JOB_INTERVAL)
            v
+-------------------------------------+
|             JOB QUEUE              |
+-------------------------------------+
     | job1 | job2 | job3 | ...
     v
Worker: chạy tuần tự → AE/OCSVM/XGB → save model



# How to use this archirtecture as a service

- In Service-based_IL run these following command on your Linux System
> sudo chmod +x install.sh
> sudo ./install.sh


# CÁC CÔNG PHÁP FIX LỖI/BẾ TẮC
### PIP REQUIREMENTS
Dùng luôn pigar
> pigar generate

### SERVICE
while read p; do
  echo "Installing $p"
  pip install "$p" || echo "❌ Failed: $p"
done < requirements.txt


Việc sử dụng git rebase thay vì git merge khi cập nhật code từ server giúp lịch sử commit của bạn luôn là một đường thẳng, sạch sẽ và dễ theo dõi hơn.Dưới đây là quy trình chuẩn để thực hiện việc này mà không gây xung đột (conflict) lung tung hay làm hỏng repo:Quy trình 4 bước chuẩn "vàng"Giả sử bạn đang làm việc trên branch feature-abc.1. Commit công việc hiện tạiTrước khi kéo code mới về, hãy đảm bảo bạn đã commit mọi thay đổi đang làm dở.Bashgit add .
git commit -m "Tính năng đang làm: mô tả ngắn gọn"
Lưu ý: Nếu chưa muốn commit vì code chưa xong, bạn có thể dùng git stash để tạm cất đi.2. Cập nhật code mới nhất từ serverThay vì dùng git pull (thường sẽ tự động tạo một merge commit), hãy dùng flag --rebase:Bashgit pull --rebase origin main
(Thay main bằng tên branch chính của dự án bạn, ví dụ develop hoặc master).3. Giải quyết xung đột (Nếu có)Nếu có file bị trùng lặp chỉnh sửa, Git sẽ dừng lại và báo lỗi. Bạn cần:Mở file bị lỗi lên và sửa lại cho đúng.Sau khi sửa xong, gõ: git add <tên_file_đã_sửa>.Tiếp tục quá trình rebase bằng lệnh:Bashgit rebase --continue
Lưu ý: Tuyệt đối không dùng git commit trong lúc này.4. Push code lên serverSau khi rebase xong, branch của bạn bây giờ đã nằm "đè" lên trên những commit mới nhất của main.Nếu bạn chưa bao giờ push branch này lên: Chỉ cần git push origin feature-abc.Nếu bạn đã từng push branch này lên trước đó: Bạn buộc phải dùng lệnh "ép" vì lịch sử đã thay đổi:Bashgit push --force-with-lease
(Dùng --force-with-lease an toàn hơn --force vì nó sẽ không đè nếu có ai đó khác cũng vừa push vào branch của bạn).Tại sao nên dùng cách này?Đặc điểmGit Merge (Mặc định)Git RebaseLịch sửChằng chịt, có nhiều nhánh rẽLà một đường thẳng tắpMerge CommitTạo ra commit rác kiểu "Merge branch..."Không tạo commit rácTính an toànRất an toàn, dễ hiểuCần hiểu rõ mình đang làm gìMột quy tắc "sống còn" khi dùng RebaseChỉ rebase trên những branch cá nhân (feature branch của riêng bạn). Tuyệt đối không bao giờ rebase trên các branch chung như main hay develop vì nó sẽ làm thay đổi lịch sử của tất cả mọi người trong team, gây ra thảm họa về quản lý code.Bạn có đang gặp phải thông báo lỗi cụ thể nào khi thực hiện các bước trên không? Tôi có thể giải thích chi tiết lỗi đó cho bạn.



# Công pháp pip
pip list --format=freeze > requirements.txt


# Nếu chạy service lỗi tại ký tự ^M$ của windows 
sed -i 's/\r$//' install_service.sh
> Sợ thì tạo backup
sed -i.bak 's/\r$//' install_service.sh



sudo ip link add veth0 type veth peer name veth1
sudo ip link set veth0 up
sudo ip link set veth1 up


ip link show veth0
ip link show veth1

tcpreplay -i eth0 attack.pcap

sudo tcpreplay -i veth0 --tcpedit \
  --enet-smac=d6:fb:b3:48:3d:79 \
  --enet-dmac=ff:ff:ff:ff:ff:ff \
  Recon_HostDiscovery.pcap

tcpreplay -i veth0 --topspeed Recon-HostDiscovery.pcap

tcpreplay -i veth0 --mbps=50 Recon-HostDiscovery.pcap

tcpreplay -i veth0 --mtu-trunc --topspeed Recon-HostDiscovery.pcap


sudo ip link set veth0 mtu 36000
sudo ip link set veth1 mtu 36000

🟢 Test IDS rule-based
tcpreplay -i veth0 \
  --pps=3000 \
  --stats=10 \
  Recon-fixed.pcap

🟢 Test ML-based IDS
tcpreplay -i veth0 \
  --pps=1000 \
  --stats=10 \
  Recon-fixed.pcap

sudo  tcpreplay -i veth1 \
  --pps=33000 \
  --stats=10 \
 DDoS-ICMP_Flood.pcap

sudo tcpreplay --pps=33000 --loop=0 --limit=1000000 -i veth1  DDoS-ICMP_Flood.pcap

  tcpreplay -i veth1 \
  --pps=1000 \
  --stats=10 \
  Recon-fixed.pcap

🔴 Stress test
tcpreplay -i veth0 \
  --pps=20000 \
  --stats=5 \
  Recon-fixed.pcap

cp -r ./src /opt/incremental_ids
cp -r systemd_service_file/* /etc/systemd/system
systemctl daemon-reexec
systemctl daemon-reload
systemctl enable ids_il.timer
systemctl start ids_il.timer



thử DDoS với số lượng mẫu khác nhau
tài nguyên, thời gian, phản hồi, biểu đồ
So sánh kịch bản

thêm chữ dưới Scenario, chỉnh sủa hình


video demo
slide 15, chi tiết, hight light animation

vẽ lại biểu đồ workflow sys ids

ICON
⟳
✔
⟳✔⏱︎⏲︎





Linux/Mac:
bash
# Phương pháp 1: jps (Java Virtual Machine Process Status Tool)
jps -l
# Output: 12345 com.example.MyApp

# Phương pháp 2: ps + grep
ps aux | grep java
ps -ef | grep java

# Phương pháp 3: pgrep
pgrep -f java
pgrep -f MyApp
Windows:
bash
# Command Prompt
jps -l
tasklist | findstr java

# PowerShell
Get-Process java
Get-Process | Where-Object {$_.ProcessName -like "*java*"}
2. JStack - Tool mạnh nhất để phân tích thread
bash
# Lấy thread dump
jstack <pid> > thread_dump.txt

# Lấy multiple dumps để phân tích
for i in {1..5}; do jstack <pid> > thread_dump_$i.txt; sleep 2; done

# Với container Docker
docker exec <container_id> jstack <pid> > thread_dump.txt
3. JConsole - GUI Monitoring
bash
# Start jconsole
jconsole <pid>

# Hoặc remote
jconsole hostname:port
4. VisualVM - Advanced Profiling
bash
# Download từ https://visualvm.github.io/
# Attach vào process
jvisualvm
5. Java Mission Control (JMC) - Production Grade
bash
# JDK 11+ (cần download riêng)
jmc
6. Shell script tự động phát hiện deadlock
bash
#!/bin/bash
# find_java_deadlocks.sh

PID=$1
OUTPUT_FILE="thread_analysis_$(date +%Y%m%d_%H%M%S).txt"

echo "=== Monitoring Java PID: $PID ===" | tee $OUTPUT_FILE

# Lấy 5 thread dump cách nhau 3 giây
for i in {1..5}; do
    echo -e "\n--- Thread Dump #$i at $(date) ---" | tee -a $OUTPUT_FILE
    jstack $PID | tee -a $OUTPUT_FILE
    
    # Kiểm tra deadlock
    DEADLOCK_COUNT=$(jstack $PID | grep -c "deadlock")
    if [ $DEADLOCK_COUNT -gt 0 ]; then
        echo "⚠️  DEADLOCK DETECTED! Count: $DEADLOCK_COUNT" | tee -a $OUTPUT_FILE
        jstack $PID | grep -A 20 "deadlock" | tee -a $OUTPUT_FILE
    fi
    
    # Kiểm tra blocked threads
    BLOCKED_COUNT=$(jstack $PID | grep -c "BLOCKED")
    echo "Blocked threads: $BLOCKED_COUNT" | tee -a $OUTPUT_FILE
    
    if [ $i -lt 5 ]; then
        sleep 3
    fi
done

# Phân tích top CPU threads
echo -e "\n=== Top CPU-consuming threads ===" | tee -a $OUTPUT_FILE
top -H -b -n 1 -p $PID | head -20 | tee -a $OUTPUT_FILE
7. Sử dụng Java API để detect programmatically
java
import java.lang.management.*;
import java.util.*;

public class ThreadMonitor {
    
    public static void monitorThreads() {
        ThreadMXBean threadBean = ManagementFactory.getThreadMXBean();
        
        // 1. Kiểm tra deadlock
        long[] deadlockedThreads = threadBean.findDeadlockedThreads();
        if (deadlockedThreads != null && deadlockedThreads.length > 0) {
            System.err.println("Deadlock detected!");
            for (long threadId : deadlockedThreads) {
                ThreadInfo info = threadBean.getThreadInfo(threadId);
                System.err.println("Deadlocked thread: " + info.getThreadName());
                System.err.println("Lock: " + info.getLockName());
                System.err.println("Lock owner: " + info.getLockOwnerName());
            }
        }
        
        // 2. Lấy tất cả threads
        ThreadInfo[] allThreads = threadBean.dumpAllThreads(true, true);
        
        // 3. Tìm blocked threads
        List<ThreadInfo> blockedThreads = new ArrayList<>();
        for (ThreadInfo thread : allThreads) {
            if (thread.getThreadState() == Thread.State.BLOCKED) {
                blockedThreads.add(thread);
            }
        }
        
        if (!blockedThreads.isEmpty()) {
            System.out.println("\n=== BLOCKED THREADS ===");
            for (ThreadInfo thread : blockedThreads) {
                printThreadInfo(thread);
            }
        }
        
        // 4. CPU time per thread
        System.out.println("\n=== THREAD CPU TIME ===");
        for (ThreadInfo thread : allThreads) {
            long cpuTime = threadBean.getThreadCpuTime(thread.getThreadId());
            long userTime = threadBean.getThreadUserTime(thread.getThreadId());
            if (cpuTime > 1000000000L) { // > 1 second
                System.out.printf("%s - CPU: %.2fs, User: %.2fs\n",
                    thread.getThreadName(),
                    cpuTime / 1e9,
                    userTime / 1e9);
            }
        }
    }
    
    private static void printThreadInfo(ThreadInfo thread) {
        System.out.println("Thread: " + thread.getThreadName());
        System.out.println("State: " + thread.getThreadState());
        System.out.println("Blocked on: " + thread.getLockName());
        System.out.println("Blocked by: " + thread.getLockOwnerName());
        System.out.println("Stack trace:");
        for (StackTraceElement element : thread.getStackTrace()) {
            System.out.println("  " + element);
        }
        System.out.println();
    }
    
    // Scheduled monitoring
    public static void startMonitoring(int intervalSeconds) {
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
        scheduler.scheduleAtFixedRate(
            ThreadMonitor::monitorThreads,
            0, intervalSeconds, TimeUnit.SECONDS
        );
    }
}
8. Arthas - Tool mạnh nhất cho production
bash
# Start Arthas
java -jar arthas-boot.jar

# Attach vào process
[arthas@1]$ dashboard  # Real-time dashboard
[arthas@1]$ thread     # List all threads
[arthas@1]$ thread -n 3  # Top 3 busy threads
[arthas@1]$ thread -b    # Find blocked threads
[arthas@1]$ thread <tid> # Check specific thread
[arthas@1]$ thread --state BLOCKED  # Filter by state
9. Perf - Linux Performance Analysis
bash
# Lấy flame graph cho Java
git clone https://github.com/brendangregg/FlameGraph.git

# Profiling với perf
perf record -F 99 -p <pid> -g -- sleep 30
perf script | ./FlameGraph/stackcollapse-perf.pl | ./FlameGraph/flamegraph.pl > flame.svg
10. Async-Profiler
bash
# Download từ https://github.com/jvm-profiling-tools/async-profiler

# CPU profiling
./profiler.sh -d 30 -f profile.svg <pid>

# Allocation profiling
./profiler.sh -d 30 -e alloc -f alloc.svg <pid>

# Lock profiling
./profiler.sh -d 30 -e lock -f lock.svg <pid>
11. Đọc và phân tích thread dump
Pattern phát hiện vấn đề:
Deadlock pattern:

text
Found one Java-level deadlock:
"Thread-1":
  waiting to lock monitor 0x00007f8b4800a2b8 (object 0x00000000ff1e8d70, a java.lang.Object),
  which is held by "Thread-2"
"Thread-2":
  waiting to lock monitor 0x00007f8b4800a2c8 (object 0x00000000ff1e8d80, a java.lang.Object),
  which is held by "Thread-1"
Blocked thread pattern:

text
"pool-1-thread-3" #17 prio=5 os_prio=0 tid=0x00007f8b4c0c3000 nid=0x4e3f waiting for monitor entry [0x00007f8b2a7f1000]
   java.lang.Thread.State: BLOCKED (on object monitor)
   at com.example.Resource.process()
   - waiting to lock <0x00000000ff1e8d70> (a com.example.Resource)
High CPU thread pattern:

text
"VM Thread" os_prio=0 tid=0x00007f8b4800a000 nid=0x4e38 runnable
12. Script tự động hóa monitoring
bash
#!/bin/bash
# auto_thread_monitor.sh

if [ -z "$1" ]; then
    echo "Usage: $0 <java_process_name>"
    exit 1
fi

PROCESS_NAME=$1
PID=$(jps -l | grep "$PROCESS_NAME" | awk '{print $1}')

if [ -z "$PID" ]; then
    echo "Process $PROCESS_NAME not found"
    exit 1
fi

echo "Monitoring PID: $PID"

# Continuous monitoring
while true; do
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    
    # Check blocked threads
    BLOCKED_COUNT=$(jstack $PID | grep "java.lang.Thread.State: BLOCKED" | wc -l)
    
    if [ $BLOCKED_COUNT -gt 0 ]; then
        echo "[$TIMESTAMP] ⚠️  $BLOCKED_COUNT blocked threads detected"
        
        # Take thread dump
        jstack $PID > thread_dump_$(date +%s).txt
        
        # Get top CPU threads
        top -H -b -n 1 -p $PID | grep -A 10 "PID" > cpu_usage_$(date +%s).txt
    fi
    
    sleep 10
done
Các công cụ khuyên dùng:
Development/Testing: VisualVM, JConsole

Production Diagnostics: Arthas, Async-Profiler

Deep Analysis: JMC, thread dump analysis

Performance Benchmark: JMH + Async-Profiler

Quick Checklist khi thread bị nghẽn:
jps -l → Tìm PID

top -H -p <pid> → Thread nào CPU cao

jstack <pid> → Tìm BLOCKED/WAITING threads

jstack <pid> | grep -A 30 "deadlock" → Check deadlock

arthas → Real-time analysis

Tùy vào môi trường (dev/prod) và quyền truy cập mà chọn tool phù hợp!


# KỊCH BẢN

# Tạo cặp veth0 và veth1 (tự động kết nối với nhau)
sudo ip link add veth0 type veth peer name veth1

# Đưa cả 2 interface lên
sudo ip link set veth0 up
sudo ip link set veth1 up

# Gán IP để test ping
sudo ip addr add 192.168.0.1/24 dev veth0
sudo ip addr add 192.168.0.2/24 dev veth1

# Kiểm tra
ip addr show veth0
ip addr show veth1


sudo hping3 --rand-source -c 100000 -i u300 -q 10.0.0.2