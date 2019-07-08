from Predictor import main as pm
from Endecoder import main as em
from Processing import main as cm
import os, warnings
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if __name__ == '__main__':
    if not os.path.exists('train.tags.en-zh.zh.deletehtml.segement.id'):
        cm()
    if not os.path.exists('checkpoint'):
        em()
    pm()
