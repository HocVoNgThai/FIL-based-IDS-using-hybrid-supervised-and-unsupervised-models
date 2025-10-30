from sklearn.preprocessing import StandardScaler
from 

class IncrementalPipeline:
    def __init__(self, input_dim, DEVICE):
        self.input_dim = input_dim
        # models (to be trained offline)
        self.bae = AE(input_dim).to(DEVICE)  # bottleneck AE
        self.ae = AE(input_dim).to(DEVICE)   # auxiliary AE for recon-error detection
        self.scaler = StandardScaler()
        self.ocsvm = None
        self.xgb = None

        # Buffers
        self.unknown_buffer = deque(maxlen=BUFFER_MAX)
        self.replay_buffer = deque(maxlen=REPLAY_SIZE)  # for replay during fine-tune
        # bookkeeping for EWC
        self.ewc_bae = None
        self.ewc_ae = None
